# The Workaholic-Willy guide

These guides take you from an empty directory to a robot picking an object, in order. Each one assumes
only what the ones before it established, so read them in sequence the first time; the short path to a
first run is the [Quickstart](../Quickstart.md), and the same ground as short programs is
[`examples/`](../../examples/README.md).

Everything here is written against the code as it is. Where a capability is built but not running,
the guide says so and says why. See [What is on, and what is only built](#what-is-on-and-what-is-only-built).

## Read in this order

| # | Guide | What you have at the end |
|---|-------|--------------------------|
| 1 | [Building the configuration](01-configuration.md) | Your cell's config tree, validating, and how to ask what a key means, where it was set and what the cell decided. |
| 2 | [The perception models](02-models.md) | Detection and segmentation chosen by config, weights on disk, loading and detecting, all settled before you touch a robot. |
| 3 | [Hand-eye calibration: the derivations](03-calibration.md) | Why it is `AX = XB`, what each mounting solves for, what the residual measures, how the transform reaches a grasp. |
| 4 | [Robot, drivers and the safety stack](04-robot-and-safety.md) | A driver connected, an end-effector declared, every guard understood in its order, motion routed through the planner. |
| 5 | [Assembling and running the pick loop](05-pick-loop.md) | A pick executing, and a map of which advanced behaviour a config key can switch on and which needs code. |
| 6 | [Driving a gripper](06-grippers.md) | Your gripper moving: its driver, the read-only steps that settle polarity and wiring, the traps each protocol carries. |

Guide 1 needs no hardware at all, and most of guide 2 needs none either: it gets you to a perception
stack that loads and detects on a CPU. Guides 3 and 4 are read at a desk as well; they explain what a
cell has to settle before it moves. The first simulator boot is at the end of guide 2, and every step
that boots one also assumes the two motion sidecars are installed, because the simulator refuses to
boot without them ([`ext_deps/README.md`](../../ext_deps/README.md)).

`WILLY_PROFILE` is sticky: it lives for the whole shell session, and every tree load honours it,
including in the next guide you read. Guide 2 asks you to set it to `sim`. Unset it again before the
later guides, whose transcripts are against the base tree, with `Remove-Item Env:\WILLY_PROFILE` on
PowerShell or `unset WILLY_PROFILE` on a POSIX shell. The non-sticky alternative is `--profile`,
passed after the subcommand.

## Or jump to what you need

| I want to | Go to |
|-----------|-------|
| understand how the config files fit together | [01](01-configuration.md), the mental model |
| find a config key without grepping YAML | [01](01-configuration.md), `where` / `explain` / `decisions` |
| know what this cell actually decided | [01](01-configuration.md), `python -m src.config decisions` |
| set up a second robot model or a different camera rig | [01](01-configuration.md), profiles and overlays |
| turn on a feature that is off by default | [01](01-configuration.md), then check [05](05-pick-loop.md), because config alone is not always enough |
| understand a validation error | [01](01-configuration.md), reading a validation failure |
| install the right requirements for my machine | [02](02-models.md), install |
| get the model weights a fresh checkout does not ship | [02](02-models.md), the weights |
| check the perception stack before booting the simulator | [02](02-models.md), proving the stack |
| understand what a hand-eye calibration is solving | [03](03-calibration.md), why it is `AX = XB` |
| decide whether a calibration result is good enough | [03](03-calibration.md), what the residual measures |
| run the calibration session at the bench | [docs/calibration-setup.md](../calibration-setup.md), and `python -m src.robot.execution.real_cell.calibrate` |
| know which drivers are validated on real hardware | [04](04-robot-and-safety.md), the drivers |
| understand exactly what refuses a motion, and in what order | [04](04-robot-and-safety.md), the safety preflight |
| choose a self-collision backend | [04](04-robot-and-safety.md), the two backends and what each is worth |
| prove the planner and the collision engine are installed | [04](04-robot-and-safety.md), and `python -m src.robot.safety.planning --check` |
| work out which gripper driver I need | [06](06-grippers.md), which driver is yours |
| measure which pin closes my jaws | [06](06-grippers.md), and `python -m src.robot.drivers.ur --measure PIN=VALUE` |
| fit a gripper this repository never shipped | [docs/runbooks/your_own_gripper.md](../runbooks/your_own_gripper.md) |
| find out why the connect refuses a hand that could not be built | [06](06-grippers.md), a substituted gripper |
| launch a simulator runner without corrupting its log | [05](05-pick-loop.md), the `cmd /c` rule |
| see a pick happen, from a cold box | [05](05-pick-loop.md) |
| clear a whole bin rather than one object | [05](05-pick-loop.md), the bin-picking orchestrator |
| turn on record logging for the soak, KPI and learning tools | [05](05-pick-loop.md), record logging |
| work out why nothing was picked | [05](05-pick-loop.md), troubleshooting |
| tell a wrist camera where to look from | [05](05-pick-loop.md), a wrist camera looks around, and [01](01-configuration.md) for `robot.look_joint_positions_deg` |
| grip only once both jaw contact faces were seen | [05](05-pick-loop.md), `both_faces` |
| let a dense pick skip a failed part, or push a boxed-in one | [05](05-pick-loop.md), recovery, and [grasping-config-reference.md](../grasping-config-reference.md) |
| pick, place and return in one call, once or until empty | [05](05-pick-loop.md), the console's task |
| make a pick faster without losing what it picks, and see which switch the owner's cell runs | [05](05-pick-loop.md), where a pick's seconds go |
| find out where a cell's seconds went, stage by stage | [scripts/cell/](../../scripts/cell/README.md) |
| give the cell a task from a browser, by text or by voice | [api/README.md](../../api/README.md) and [frontend/README.md](../../frontend/README.md) |
| halt the arm from the console, and know what that is not | [04](04-robot-and-safety.md), halt now, the emergency stop, and the way back |
| teach a place pose by hand and let a task use it | [01](01-configuration.md), `robot.named_poses` and `robot.default_place_pose` |
| answer a toggle hand's jaws question in the browser | [06](06-grippers.md), the jaws question in the browser |
| bring the console to a physical cell | [docs/runbooks/console_at_the_cell.md](../runbooks/console_at_the_cell.md) |

## What is on, and what is only built

Two facts the guides repeat, because believing otherwise costs a day.

**The default pick is open-loop.** Perceive, generate and score candidates on a deterministic
geometric rank, run the safety preflight and IK, move and close, log. The decision gate, recovery,
fixed-camera fusion, feasibility and occlusion scoring, clutter ordering, the learned success model and
the reinforcement-learning layer are all built and all default to off. One thing needs no switch: **a
wrist camera looks around** whenever a pick hands it looks, fuses them and stops as soon as the grasp is
safe ([05](05-pick-loop.md)). Check any claim of that shape against the tree rather than against a
document:

```bash
python -c "from willy import load_tree; g = load_tree().robot.grasping; print({n: getattr(g, n).enabled for n in ('fusion', 'decision', 'recovery', 'success_model')})"
```

**There are two composition paths, and the simulator mostly does not use the real one.**
`AutonomousGraspService.from_robot_config()` is the config-driven boot path a physical cell takes,
reached through `build_real_cell` and `build_rehearsal_cell` in
[`src/robot/execution/autonomous_grasp/cells.py`](../../src/robot/execution/autonomous_grasp/README.md),
and through `Cell` from Python and [`python -m src.robot.execution.real_cell`](../../src/robot/execution/real_cell/README.md)
from a shell. Most simulator runners build through `from_components` instead, because the simulated
gripper is not in the gripper registry and needs the arm's session. That leaves `effective_config=None`
and silences the config-driven overlays, so those runners re-enable them in runner code. Two runners
can take the config path, `run_multiview_pick` by default and `run_eih_pick` on request. The
consequence is in [05](05-pick-loop.md).

A file staying silent about a setting means the schema default is in force, not that the feature
does not exist. `python -m src.config where <substring>` searches the schema, so it finds the keys no
YAML mentions. It lists at most 40 keys per tier; pass `--limit 500` for the complete listing, and
`--tier <safety|site|tuned|advanced|all>` to narrow it.

## What "verified" means on these pages

Every command printed here was run against this repository, and every path and config key named here
was resolved against it. What is not promised is a number. The pick rates a cell reaches depend on
its optics, its objects and its gripper, so the guides state the rule a gate applies rather than the
score somebody else's run got. Where a guide does report a result, it says which of four levels of
evidence stands behind it, and they mean exactly this:

| Level | Means |
|---|---|
| **run on a physical cell** | a real arm, hand or camera driven by this code at a cell; the page names what ran, and no pick rate from it is kept here |
| **measured in simulation** | Isaac Sim, on the box, with numbers |
| **measured against real controller software** | a real protocol or a real controller, no physical motion |
| **never touched hardware** | code complete, behaviour unproven |

**Where it has run.** A **UR10 (CB3)** with a **D415 on the wrist** and a **Hand-E switched over one tool
output** (`jaw_io`, `single_toggle`) has run this code from a powered-off arm to camera picks: the
connect, home and moves planned with cuRobo (example 03), the wrist camera's calibration by hand in
freedrive against one ArUco marker (examples 09 and 10), detection and segmentation on its frames, and
camera picks through `PickRun` (example 12) and the `Locator` (example 13). Two gripper drivers were
measured with a UR: `jaw_io` and the Robotiq driver over its URCap socket, whose port 63352 this cell
refused, so its Hand-E runs as `jaw_io`; OnRobot and suction never touched hardware. **Not run there
yet:** the wrist looks of 2026-09-29, their generated view and the push; the operator console and the
API, whose task, halt now, way back after a stop and jaws question were measured against URSim CB3 (a UR10,
2026-10-01 and 2026-10-02) and nowhere else; and the deep grasp network, which was never trained. A few
package pages write this level as
*measured on a UR10*, *measured against a UR10*, *measured against UR-Robot* or *measured on a real
camera setup*.

Two consequences worth carrying between guides:

- **The simulator gate and the real-cell runner apply different rules.** A simulator campaign passes
  when `int(pass_fraction * runs)` picks pass, with `pass_fraction` set to `0.8` in the tree, and a
  run counts as passing only when `pick()` reports success and, independently, the object's measured
  world-Z rose by at least `robot.sim.gate.lift_threshold_mm`, which the tree sets to `50.0`. The
  real-cell runner requires every attempt to succeed and takes the pick service's word for each, which
  is why it prints its rule beside its verdict. From Python, `PassRule(fraction=..., confirm=...)` sets
  another fraction and a check of your own on each attempt
  ([`examples/real_robot/12_pick_with_the_camera.py`](../../examples/real_robot/12_pick_with_the_camera.py)).
- **A failing run says little about why.** Each runner prints one line per pick, `succeeded`,
  `lift_mm` and `passed`, and then a gate line. `run_m2_pick` adds a reason beside its rate: each run
  that did not pick, with the report's failure summary and the motion's message. Attributing a
  failure beyond that needs record logging, which is `grasping.record_log_path` and off by default.

## Related documents

These are not part of the walkthrough, but the guides link into them.

| Document | For |
|----------|-----|
| [docs/Quickstart.md](../Quickstart.md) | the short path: a pick on a dummy arm, the desk check, a profile |
| [examples/README.md](../../examples/README.md) | the same ground as short programs through `from willy import ...`: your cell, simulation, offline work |
| [willy/README.md](../../willy/README.md) | every name `from willy import ...` gives you, and the example that shows it |
| [docs/cli.md](../cli.md) | every command line, by topic, and the runbook that uses it |
| [docs/runbooks/](../runbooks/) | bench procedures: cell bring-up, a first real pick, the console at the cell, your own gripper, the Hand-E, UR arms, corpus builds, training |
| [docs/calibration-setup.md](../calibration-setup.md) | the bench procedure: print the board, run the sweep, wire the artifact in. [03](03-calibration.md) is the concepts, this is the session |
| [docs/isaac-ready.md](../isaac-ready.md) | the cold-box simulator checklist: the standalone interpreter, the asset pack, the logging pattern and the gate criterion |
| [ext_deps/README.md](../../ext_deps/README.md) | installing the planner and collision sidecars with `scripts/ext_deps/install.ps1` |
| [docs/safety-math.md](../safety-math.md) | the derivations behind the safety bounds |
| [docs/grasping-math.md](../grasping-math.md) | the derivations behind the grasp itself, from prompt to point cloud to grasp pose |
| [docs/deep-network.md](../deep-network.md) | the learned grasp network layer by layer, how it learns, and what it can and cannot add to the analytic generator |
| [docs/grasping-config-reference.md](../grasping-config-reference.md) | every `robot.grasping` block, and which grasp mode it can fire in |
| [src/willy_sim/README.md](../../src/willy_sim/README.md) | the simulator harness behind every simulation step in these guides |
