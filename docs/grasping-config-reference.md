# The `robot.grasping` config, and why enabling a block often does nothing

[Workaholic-Willy](../README.md) | siblings: [`grasping-math.md`](grasping-math.md),
[`safety-math.md`](safety-math.md)

Read sections 2 and 3 before you enable or measure any `robot.grasping` block. Most blocks cannot
fire in most grasp modes, and four of them ship the value that would make them matter set to zero.
Both facts turn an honest measurement into a confident wrong answer.

This page is not a key reference. [`config/all_keys/`](../config/all_keys/) is: a complete tree in
which every key the schema accepts is written out with what it does, its default and its legal
values. It validates on its own, so it is checkable rather than merely documentation, and nothing
loads it by default:

```bash
python -m src.config --data config/all_keys                       # the reference tree validates
python -m src.config explain robot.grasping.fusion.enabled        # type, default, tier, who set it
python -m src.config where ordering --tier all                    # search the schema, not the files
python -m src.config decisions                                    # only what differs from default
```

What you get here is the part a generated index cannot give you: which blocks reach the pick path at
all, and what goes wrong when you assume they do.

## 1. How the config reaches the stack

`AutonomousGraspService.from_robot_config` reads `robot.grasping`, builds an
`EffectiveGraspingConfig`, and installs a set of carriers on the orchestrator. That is the path a
real cell takes, and it is the only path that honours the config.

`from_components` is the other constructor, and it leaves `effective_config` as `None`. Almost every
`grasping.*` block is then silent, because nothing built it. Several simulation runners use this
constructor and re-enable what they want in runner code, per command-line flag. So a simulation run
can have a decision engine while `grasping.decision.enabled` is false, and turning that key on
changes nothing. Section 3.2 is the same trap seen from the other side.

## 2. Grasp modes, the gate in front of every block

`robot.grasping.default_mode`, or the service's `mode=` argument, selects a behaviour profile. The
profiles live in `src/robot/execution/autonomous_grasp/config.py`, in `_PROFILES`, and that table is the
authority.

| Mode | Sampling | Refine | Verify | Recovery actions allowed |
|---|---|---|---|---|
| `easy` | single object | no | no | none, and that is a guarantee rather than a default |
| `auto` | auto | no | no | `rescan`, `next_viewpoint` |
| `dense_clutter` | dense point cloud | no | no | `rescan`, `next_viewpoint` |
| `closed_loop` | auto | yes | yes | `rescan`, `next_viewpoint` |
| `dense_autonomous` | dense point cloud | yes | yes | `rescan`, `next_viewpoint`, `nudge_target` |

`config/robot/robot.yaml` ships `auto`. The simulation runners default to `easy`.

**This is why a block measurement can be worth nothing.** Each mode-gated block declares its own
`apply_modes`, and `build_effective_config` enforces that filter. `easy` appears in the `apply_modes`
of exactly one block, `success_model`.

| Block | Its `apply_modes` default | Effect of a mode outside it |
|---|---|---|
| `feasibility` | `auto`, `dense_clutter`, `dense_autonomous` | every key collapses to its off value |
| `occlusion` | `auto`, `dense_clutter`, `dense_autonomous` | every key collapses to its off value |
| `ordering` | `auto`, `dense_clutter`, `dense_autonomous` | every key collapses to its off value |
| `recovery` | `auto`, `dense_clutter`, `dense_autonomous` | every key collapses to its off value |
| `uncertainty` | `auto`, `dense_clutter`, `dense_autonomous` | the runtime consults the list; the snapshot carries it |
| `success_model` | `easy`, `auto`, `dense_clutter`, `dense_autonomous` | `enabled` reads false |
| `approach_validation` | `dense_clutter`, `dense_autonomous` | `enabled` reads false |
| `fusion.commit_policy` | `auto`, `dense_clutter`, `dense_autonomous` | the gate is skipped |

Note that four of those lists exclude `closed_loop` as well as `easy`, which is easy to miss when
`closed_loop` is the mode you reached for precisely because you wanted more behaviour.

Being built is not the same as being acted on. Three blocks are built in every mode and run in
almost none:

* `decision` fires only when the effective mode is not `easy`. The check is in `service.py`, on the
  path into `_pick_with_decision`.
* `closed_loop` and `verification` are gated on the profile's `refinement_enabled` and
  `verification_enabled`, so they run in `closed_loop` and `dense_autonomous` only.
* `approach_validation` installs its carrier in every mode and fires only in the dense modes. The
  gate is in `pick_loop.py`.

That last shape, a carrier that is set and never read, is the one the wiring guard cannot see. The
guard's claim is that a flag reaches a runtime carrier, and `EffectiveGraspingConfig` is a runtime
carrier. Reaching a carrier and the pick path acting on it are different claims, and knowing that
limit is part of using the guard.

## 3. The four traps

Each one turns a measurement into a lie that reads as good news.

### 3.1 The zero-weight trap

Four blocks ship their operative value at zero or empty. Setting `enabled: true` and nothing else
computes the signal and then multiplies it by nothing.

| What you must also set | Ships as | What happens if you do not |
|---|---|---|
| `feasibility.weight` | `0.0` | the signal is computed and contributes nothing to the rank |
| `ordering.unlock_weight` | `0.0` | the blocker graph is built and never reorders anything |
| `uncertainty.ranking_penalty_weight` | `0.0` | the ranking half is inert, though the fail-closed half still fires |
| `recovery.allowed_actions` | `()` | the orchestrator is permitted to do nothing |

### 3.2 The constructor wins

`build_subpolicies` prefers a constructor argument over the config block: it materialises a
refinement, verification or recovery policy only where the caller passed none. The simulation
runners hand `from_robot_config` a hand-built `DecisionEngine` in `auto`, and a refiner plus a
verifier in `closed_loop`. So `--boot config --grasp-mode auto` runs a decision engine that came out
of the runner, and toggling `grasping.decision.enabled` changes nothing at all. Pass
`--config-subpolicies` to suppress the runner's and let the config own them; it requires
`--boot config`, and refuses otherwise rather than silently doing nothing.

### 3.3 Two things are called "mode"

On `run_multiview_pick`, `--mode` selects the camera set: `eth1`, `eth2`, `eth3` or `sides`. The
grasp mode is `--grasp-mode`. They are unrelated, and the logs call both of them mode.

### 3.4 Two blocks are called "recovery"

`grasping.recovery` is the orchestrator: when to recover, and within what budget.
`grasping.dense_recovery` is the strategy: what recovering means, one of `active_perception`,
`next_target` or `none`. Different consumers, different allow-lists. Wiring one of them gives you
half a recovery path.

## 4. What each block is for

Keys and defaults are in [`config/all_keys/`](../config/all_keys/). This is the one-line purpose and
the mode gate. Every block in this table ships `enabled: false`, except `occlusion`, which has
`directional_enabled` and `hard_reject_enabled` instead and ships both false.

| Block | What it does | Fires in |
|---|---|---|
| `decision` | the fail-closed gate: accept, escalate or refuse | every mode except `easy` |
| `closed_loop` | re-perceive and refine before the final descent | `closed_loop`, `dense_autonomous` |
| `verification` | check after the grasp whether anything was picked | `closed_loop`, `dense_autonomous` |
| `dense_recovery` | what recovering means, as a strategy | dense modes |
| `recovery` | when to recover and within what budget | not `easy` or `closed_loop` |
| `feasibility` | reachability-aware re-ranking | not `easy` or `closed_loop` |
| `occlusion` | directional corridor analysis | not `easy` or `closed_loop` |
| `ordering` | bin-clearing order from a blocker graph | not `easy` or `closed_loop` |
| `uncertainty` | a fused uncertainty score, with a fail-closed half and a ranking half | not `easy` or `closed_loop` |
| `success_model` | the learned success predictor | every mode except `closed_loop`, and the only one `easy` reaches |
| `performance` | latency budgets per stage | every mode |
| `approach_validation` | the swept-volume check along the approach | carrier everywhere, fires in dense modes |
| `fusion` | the multi-view voxel substrate and the commit gate | every mode |
| `deep_ranker` | the learned ranker in shadow: it scores, the telemetry records, the order never changes | every mode |

`ordering` has a second precondition that no key expresses: it can only act where the loop is free
to choose which object goes first. With a target label the choice is made before ordering is
consulted, so the block is structurally inert.

The always-on tunings are a separate family and are not gated by mode:

* `calculator` picks the generator, `geometric` or `deep`. `deep` with no readable artifact refuses
  to build the cell rather than falling back, and no trained weights ship here.
* `deep_generator` sizes and points that learned generator; it is inert while `calculator` is
  `geometric`.
* `geometry.stage` chooses which stage proposes candidates, `support_footprint` or `silhouette`.
* `support` says where the world's floor is, and the bin or tray is `support.container`, not
  `grasping.container`.
* `fusion.geometry` is per-object multi-camera geometry fusion, a different thing from `fusion`.
* `gripper_geometry` is the collision envelope the planner checks, parallel jaw or suction cup.
* `watchdog` has a `mode` rather than an `enabled`, and it ships `shadow`.
* `isotropic_radial_closing`, `max_attempts` and `record_log_path` sit at the top of the block.

## 5. How to measure a block

One block per run, each against a baseline in the same mode with the same flags:

```bash
# baseline for this mode
python -m src.willy_sim.run_multiview_pick --runs 5 --boot config \
    --scene tall --rendered-depth --grasp-mode auto --config-subpolicies

# the same thing with exactly one block layered on
python -m src.willy_sim.run_multiview_pick --runs 5 --boot config \
    --scene tall --rendered-depth --grasp-mode auto --config-subpolicies \
    --profile decision
```

`--profile decision` stacks [`config/robot/robot.decision.yaml`](../config/robot/robot.decision.yaml)
on top of the loaded tree, which turns the gate on and nothing else, so a measured difference is
attributable to it. Seven rules:

1. Use `--boot config`, or the measurement is about the runner and not about the cell.
2. Use `--rendered-depth`, or the measurement is about a ground-truth oracle that reports the
   object's centre rather than a surface any real depth camera would deliver.
3. Use `--config-subpolicies` in `auto` and `closed_loop`, or the constructor wins. See section 3.2.
4. The mode has to be one the block can fire in. See section 2.
5. Set the block's operative value, not only `enabled: true`. See section 3.1.
6. Read the right number. For a fail-closed gate, a drop in picks is the expected result, so
   counting lift is the wrong metric for most of these blocks.
7. Gate on the log line rather than the exit code. A crashed simulation process hangs instead of
   returning, so a wall-clock timeout is not a pass.

## 6. What the schema refuses

### 6.1 The declared-unwired list

`RobotGraspingConfig.UNWIRED_SWITCHES` is a list of flags whose value reaches
`EffectiveGraspingConfig`, which is the cell's own telemetry, and is then read back by nothing.
Turning one on changes what the cell reports about itself and not one thing about what it does. A
model validator refuses to load a config that sets one, with the reason and a pointer to this
section, so an operator meets the truth where they set the value rather than in a document they may
never open.

Read the list from the code rather than from here:

```bash
python -c "from src.config.schema.robot.grasping_schema import RobotGraspingConfig as G; print(G.UNWIRED_SWITCHES)"
```

It refuses rather than warns because an operator who sets one of these gets a cell that boots, runs,
reports the flag as on, and behaves exactly as before. That is not a crash; it is a false sense of a
capability. The flags stay in the schema because the capability exists. What is missing is the wire
from the key to it, and when one is wired its entry is deleted and the block is measured.

Two rules keep the list honest, and they only mean something together. Every flag must either move
the runtime or be declared unwired and refuse to load, so a flag may not be quietly acceptable and
inert. And the unwired list may not name a switch the schema walker cannot find, so a stale
exemption cannot silently cover a flag that has since been wired.

`feasibility` carries a second warning that is not about wiring, and `builders.py` states it at the
point where the block is built. Expect it to cost picks, and enable it on that understanding: a good
overhead top-down grasp is itself near singular, meaning a high Jacobian condition number, so the
`ik_quality` sub-signal demotes exactly the grasp that is wanted. It is wired so the claim can be
re-measured through config, and it stays off in the shipped tree. Note also that its carrier is
installed only when `enabled` is true, `weight` is above zero, and the mode is in `apply_modes`, so
this block needs all three of section 2 and section 3.1 satisfied at once.

## See also

* [`config/all_keys/`](../config/all_keys/), the generated reference tree, every key with its
  default
* [`src/config/`](../src/config/README.md), how profiles layer and how to ask the tree about itself
* [`grasping-math.md`](grasping-math.md), the formulas these blocks evaluate
* [`src/robot/grasping/`](../src/robot/grasping/README.md), the package itself
* [`src/robot/execution/autonomous_grasp/`](../src/robot/execution/autonomous_grasp/README.md), the
  composition root that turns this config into carriers
