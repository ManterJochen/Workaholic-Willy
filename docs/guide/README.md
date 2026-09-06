# The Workaholic-Willy guide

Six guides that take you from an empty directory to a robot picking an object, in order. Each one
assumes only what the ones before it established, so read them in sequence the first time.

Everything here is written against the code as it is. Where a capability is built but not running,
the guide says so and says why. See [What is on, and what is only built](#what-is-on-and-what-is-only-built).

## Read in this order

| # | Guide | What you have at the end |
|---|-------|--------------------------|
| 1 | [Building the configuration](01-configuration.md) | A validating config tree for your own cell, and the ability to ask the config what any key means, where it was set, and what this cell actually decided. |
| 2 | [The perception models](02-models.md) | Detection and segmentation chosen by config, weights on disk, loading and detecting, all settled before you touch a robot. |
| 3 | [Hand-eye calibration: the derivations](03-calibration.md) | The ability to reason about a calibration result rather than only produce one: why the problem is `AX = XB`, what each mounting case solves for, what the residual measures, and how the transform reaches a grasp. |
| 4 | [Robot, drivers and the safety stack](04-robot-and-safety.md) | A driver connected, an end-effector declared, every safety guard understood in the order it runs, and motion routed through the planner and the exact-mesh collision engine. |
| 5 | [Assembling and running the pick loop](05-pick-loop.md) | A pick executing, and a map of which advanced behaviour a config key can switch on and which needs code. |
| 6 | [Driving a gripper](06-grippers.md) | Your gripper moving: which driver it is, the read-only steps that settle polarity and wiring before anything is commanded, and the traps each protocol carries. |

Guide 1 needs no hardware at all, and most of guide 2 needs none either: it gets you to a
perception stack that loads and detects on a CPU. Guides 3 and 4 are read at a desk as well; they
explain what a cell has to settle before it moves. The first simulator boot is at the end of guide 2,
and every step that boots one also assumes the two motion sidecars are installed, because the
simulator refuses to boot without them ([`ext_deps/README.md`](../../ext_deps/README.md)).

One session-hygiene rule that spans the whole set. `WILLY_PROFILE` is sticky: it lives for the whole
shell session and `load_config()` honours it everywhere, including in the next guide you read. Guide
2 asks you to set it to `sim`. Unset it again before the later guides, whose transcripts are against
the base tree, with `Remove-Item Env:\WILLY_PROFILE` on PowerShell or `unset WILLY_PROFILE` on a
POSIX shell. The non-sticky alternative is `--profile`, passed after the subcommand.

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
| work out which gripper driver I need | [06](06-grippers.md), the four drivers |
| measure which pin closes my jaws | [06](06-grippers.md), and `python -m src.robot.drivers.ur --measure PIN=VALUE` |
| find out why every pick reports success and nothing is held | [06](06-grippers.md), a substituted null gripper |
| launch a simulator runner without corrupting its log | [05](05-pick-loop.md), the `cmd /c` rule |
| see a pick happen, from a cold box | [05](05-pick-loop.md) |
| clear a whole bin rather than one object | [05](05-pick-loop.md), the bin-picking orchestrator |
| turn on record logging for the soak, KPI and learning tools | [05](05-pick-loop.md), record logging |
| work out why nothing was picked | [05](05-pick-loop.md), troubleshooting |

## What is on, and what is only built

Two facts the guides repeat, because believing otherwise costs a day.

**The default pick is open-loop.** Perceive, generate and score candidates on a deterministic
geometric rank, run the safety preflight and IK, move and close, log. The decision gate, closed-loop
refinement, post-grasp verification, multi-view fusion, feasibility and occlusion scoring, clutter
ordering, the learned success model and the reinforcement-learning layer are all built and all
default to off. Check any claim of that shape against the tree rather than against a document:

```bash
python -c "from src.config import load_config; g=load_config().robot.grasping; print({n: getattr(getattr(g,n),'enabled',None) for n in ('fusion','decision','closed_loop','verification','recovery','success_model')})"
```

**There are two composition paths, and the simulator mostly does not use the real one.**
`AutonomousGraspService.from_robot_config()` is the config-driven boot path a physical cell takes,
reached through `build_real_cell` and `build_rehearsal_cell` in
[`src/robot/execution/autonomous_grasp/cells.py`](../../src/robot/execution/autonomous_grasp/README.md)
and driven by [`python -m src.robot.execution.real_cell`](../../src/robot/execution/real_cell/README.md).
Most simulator runners build through `from_components` instead, because the simulated gripper is not
in the gripper registry and needs the arm's session. That leaves `effective_config=None` and
silences the config-driven overlays, so those runners re-enable them in runner code. Two runners can
take the config path, `run_multiview_pick` by default and `run_eih_pick` on request. The consequence
is in [05](05-pick-loop.md).

A file staying silent about a setting means the schema default is in force, not that the feature
does not exist. `python -m src.config where <substring>` searches the schema, so it finds the keys no
YAML mentions. It lists at most 40 keys per tier; pass `--limit 500` for the complete listing, and
`--tier <safety|site|tuned|advanced|all>` to narrow it.

## What "verified" means on these pages

Every command printed here was run against this repository, and every path and config key named here
was resolved against it. What is not promised is a number. The pick rates a cell reaches depend on
its optics, its objects and its gripper, so the guides state the rule a gate applies rather than the
score somebody else's run got.

Two consequences worth carrying between guides:

- **The simulator gate and the real-cell runner apply different rules.** A simulator campaign passes
  when `int(pass_fraction * runs)` picks pass, with `pass_fraction` set to `0.8` in the tree, and a
  run counts as passing only when `pick()` reports success and, independently, the object's measured
  world-Z rose by at least `robot.sim.gate.lift_threshold_mm`, which the tree sets to `50.0`. The
  real-cell runner requires unanimity and has no independent confirmation, which is why it prints
  its rule beside its verdict.
- **A failing run does not say why.** Each runner prints one line per pick, `succeeded`, `lift_mm`
  and `passed`, and then a gate line. None of them carries a reason. Attributing a failure needs
  record logging, which is `grasping.record_log_path` and off by default.

## Related documents

These are not part of the walkthrough, but the guides link into them.

| Document | For |
|----------|-----|
| [QUICKSTART.MD](../Quickstart.md) | the short path: pick a profile, run the gate |
| [docs/isaac-ready.md](../isaac-ready.md) | the cold-box simulator checklist: the standalone interpreter, the asset pack, the logging pattern and the gate criterion |
| [ext_deps/README.md](../../ext_deps/README.md) | the three external systems, and installing the planner and collision sidecars with `scripts/ext_deps/install.ps1` |
| [docs/calibration-setup.md](../calibration-setup.md) | the bench procedure: print the board, run the sweep, wire the artifact in. [03](03-calibration.md) is the concepts, this is the session |
| [docs/runbooks/](../runbooks/) | on-call procedures: first pick on a real cell, cell bring-up, corpus builds, training your own generator |
| [docs/safety-math.md](../safety-math.md) | the derivations behind the safety bounds |
| [docs/grasping-math.md](../grasping-math.md) | the derivations behind the grasp itself, from prompt to point cloud to grasp pose |
| [docs/grasping-config-reference.md](../grasping-config-reference.md) | every `robot.grasping` block, and which grasp mode it can fire in |
| [scripts/examples/README.md](../../scripts/examples/README.md) | the same path as seven runnable Python files, driving the library nouns rather than the command line |
| [src/willy_sim/README.md](../../src/willy_sim/README.md) | the simulator harness behind every simulation step in these guides |
