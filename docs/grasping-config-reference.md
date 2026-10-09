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

| Mode | Sampling | Recovery actions allowed |
|---|---|---|
| `easy` | single object | none, and that is a guarantee rather than a default |
| `auto` | auto | `rescan`, `next_target` |
| `dense_clutter` | dense point cloud | `rescan`, `next_target`, `nudge_target` |

`config/robot/robot.yaml` ships `auto`. The simulation runners default to `easy`.

`next_viewpoint` left both profiles on 2026-09-29, merged into `rescan`: with the viewpoint planners
gone nothing moved the camera for it, so it re-perceived exactly as `rescan` does. A
`recovery.allowed_actions` or `recovery.per_action_budget` that still names it is refused at load,
`removed on purpose: use rescan`.

`closed_loop` and `dense_autonomous`, the two modes that refined and verified, were removed on
2026-09-29 with the two-scan pre-grasp refinement they ran; no shipped config switched it on, and it
never ran on a physical arm. A tree, a preset or a call that still names one is refused with the mode
to name instead: `auto` for `closed_loop`, `dense_clutter` for `dense_autonomous`. `nudge_target`, which
only `dense_autonomous` allowed, moved to `dense_clutter`; it still needs `recovery.allowed_actions` to
name it. A `recovery.fixture` is **optional** since 2026-10-01: it only narrows where a push lands.

**The mode's profile is an outer gate** that `recovery.allowed_actions` cannot widen: `auto` allows
`rescan` and `next_target`, `dense_clutter` adds `nudge_target`, and no profile allows
`container_agitate`. `rescan` re-runs the pick, which perceives and ranks afresh. `next_target` is a
rescan that **skips the part that failed** for another part of the same label, never another object, and
moves nothing itself. `nudge_target` is **the push**, and it runs inside a wrist camera's pick attempt
only: the service's recovery loop never plans a nudge step, and one another caller's loop plans is
refused before anything moves (`refused_push_runs_in_the_pick`). `container_agitate` is refused at load unless a container is declared
(6.2). The simulator runner `run_dense_pick --g6` still widens a profile itself to film
`container_agitate`; a cell built from config never does. Section 4.1 is the recovery block, key by key.

**This is why a block measurement can be worth nothing.** Each mode-gated block declares its own
`apply_modes`, and `build_effective_config` enforces that filter. `easy` appears in the `apply_modes`
of exactly one block, `success_model`.

| Block | Its `apply_modes` default | Effect of a mode outside it |
|---|---|---|
| `feasibility` | `auto`, `dense_clutter` | every key collapses to its off value |
| `occlusion` | `auto`, `dense_clutter` | every key collapses to its off value |
| `ordering` | `auto`, `dense_clutter` | every key collapses to its off value |
| `recovery` | `auto`, `dense_clutter` | every key collapses to its off value |
| `uncertainty` | `auto`, `dense_clutter` | the runtime consults the list; the snapshot carries it |
| `success_model` | `easy`, `auto`, `dense_clutter` | `enabled` reads false |
| `approach_validation` | `dense_clutter` | `enabled` reads false |

A mode list that still names `dense_autonomous` or `closed_loop` is refused at load with the same
sentence as `default_mode`.

Being built is not the same as being acted on. Two blocks are built in every mode and run in
almost none:

* `decision` fires only when the effective mode is not `easy`. The check is in `service.py`, on the
  path into `_pick_with_decision`.
* `approach_validation` installs its carrier in every mode and fires only in `dense_clutter`. The
  gate is in `pick_loop.py`.

`verification` was the third until 2026-09-29, when it was removed: no attempt consulted the verifier
it built. The hold an attempt reports is the execution policy's own check after its close, which reads
the gripper's `is_object_detected` and `hold_evidence` (a Robotiq's gOBJ). A tree that still writes the
block is refused at load with the sentence that says so.

`recovery` runs in its modes, and its one motion, the push, runs in `dense_clutter` alone, inside a
wrist camera's pick attempt (4.1).

That last shape, a carrier that is set and never read, is the one the wiring guard cannot see. The
guard's claim is that a flag reaches a runtime carrier, and `EffectiveGraspingConfig` is a runtime
carrier. Reaching a carrier and the pick path acting on it are different claims, and knowing that
limit is part of using the guard.

## 3. The three traps

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

`build_decision_layer` prefers a constructor argument over the config block: it builds a decision
engine only where the caller passed none. The simulation runners hand
`from_robot_config` a hand-built `DecisionEngine` in `auto`. So `--boot config --grasp-mode auto` runs
a decision engine that came out
of the runner, and toggling `grasping.decision.enabled` changes nothing at all. Pass
`--config-subpolicies` to suppress the runner's and let the config own them; it requires
`--boot config`, and refuses otherwise rather than silently doing nothing.

### 3.3 Two things are called "mode"

On `run_multiview_pick`, `--mode` selects the camera set: `eth1`, `eth2`, `eth3` or `sides`. The
grasp mode is `--grasp-mode`. They are unrelated, and the logs call both of them mode.

A fourth trap left on 2026-09-29: two blocks were called "recovery". `grasping.dense_recovery`, the
strategy half, built a policy and a strategy no pick consulted, and it was removed; a tree that still
writes it is refused at load. `grasping.recovery` is the recovery a pick runs.

## 4. What each block is for

Keys and defaults are in [`config/all_keys/`](../config/all_keys/). This is the one-line purpose and
the mode gate. Every block in this table ships `enabled: false`, except `occlusion`, which has
`directional_enabled` and `hard_reject_enabled` instead and ships both false.

| Block | What it does | Fires in |
|---|---|---|
| `decision` | the fail-closed gate: accept, escalate or refuse | every mode except `easy` |
| `recovery` | when to recover and within what budget | not `easy` |
| `feasibility` | reachability-aware re-ranking | not `easy` |
| `occlusion` | directional corridor analysis | not `easy` |
| `ordering` | bin-clearing order from a blocker graph | not `easy` |
| `uncertainty` | a fused uncertainty score, with a fail-closed half and a ranking half | not `easy` |
| `success_model` | the learned success predictor | every mode, and the only one `easy` reaches |
| `performance` | latency budgets per stage | every mode |
| `approach_validation` | the swept-volume check along the approach | carrier everywhere, fires in `dense_clutter` |
| `fusion` | the fused cameras' CAMERA to BASE resolvers (`enabled`), and `fusion.geometry` below | every mode |
| `deep_ranker` | the learned ranker in shadow: it scores, the telemetry records, the order never changes | every mode |

`ordering` has a second precondition that no key expresses: it can only act where the loop is free
to choose which object goes first. With a target label the choice is made before ordering is
consulted, so the block is structurally inert.

The always-on tunings are a separate family and are not gated by mode:

* `calculator` picks the generator, `geometric` or `deep`. `deep` with no readable artifact refuses
  to build the cell rather than falling back, and no trained weights ship here. A cell also refuses
  trained weights whose proof has not passed, which today is all of them (2026-10-09).
* `deep_generator` sizes and points that learned generator; it is inert while `calculator` is
  `geometric`.
* `geometry.stage` chooses which stage proposes candidates, `support_footprint` or `silhouette`.
  `geometry.footprint_rim_mm` (**0.0**; the owner's "Ja, für Montag", 2026-10-09) cuts a rim of that many
  millimetres off every mask for the cloud the support-footprint stage builds its footprint from, and for
  nothing else: the D415 smears a part's far edge into a ramp the hull follows. Pair it with
  `geometry.inflate_mm` (**0.0**), which gives the faces back: 2.0 with 1.25 took the grasp centre on 23
  recorded cubes from 2.53 to 0.77 mm off at the median ([05](guide/05-pick-loop.md), section 2).
* `support` says where the world's floor is, and the bin or tray is `support.container`, not
  `grasping.container`.
* `fusion.geometry` is per-object multi-camera geometry fusion, and it needs `fusion.enabled` too: that
  switch builds the other cameras' CAMERA to BASE resolvers, and with it off every second view is
  dropped at the pick with a warning. Its `promote_unmatched` decides whether another camera may
  introduce an object or only confirm one the primary already found. Off, the pick sees exactly the
  primary's segmentation list, in its order. On, a part only a second camera can see becomes an
  object and is computed in that camera's lens, which is why the cell then builds one calculator per
  camera. Nothing measures how often that happens, so it ships off.
* `gripper_geometry` is the collision envelope the planner checks, parallel jaw or suction cup.
* `watchdog` has a `mode` rather than an `enabled`, and it ships `shadow`.
* `isotropic_radial_closing`, `max_attempts` and `record_log_path` sit at the top of the block.
* `scene_obstacles` (**on**) makes the calculator hold every object the camera saw beside the part as an
  obstacle, whether a prompt named it or not, with what the part stands on taken out; a part whose every
  grasp meets one is `all_collided`, the failure a blocker or a push answers
  ([05](guide/05-pick-loop.md), section 2). Off is the calculator of before, byte for byte.
* `side_approaches` (**on**, the owner's decision of 2026-10-01) lets the support-footprint search offer
  every tilt the hand fits at, scored by the room each grasp keeps, vertical on a tie, and a grasp 15
  degrees or more off vertical only through space a depth ray saw. Off, a tilted grasp is offered only where
  no vertical one fits.
* The two that come with them sit under `robot.safety.planning_world.perceived`: `support_surfaces`
  (**on**) finds what the parts stand on and holds it as solid, and `support_allowance_mm` (**2**, 0 to 10)
  is how far those solids stand over what they take out ([04](guide/04-robot-and-safety.md), 5.5). A
  library caller building `WorldBuildTuning` by hand gets neither, as before.
* `first_good_part` (**on**), `good_part_score` (**0.75**) and `fine_pass_waits` (**on**), the owner's "speed
  first" of 2026-10-08: a look's parts are computed in the order they stand apart and the first good one is
  taken, the best where none is good, and the support-footprint search's fine pass waits while another part may
  have a full result. Off, every part is computed and the best taken, as before ([05](guide/05-pick-loop.md),
  section 2).
* `workers` (**0**) and `batched_builds` (**off**) make a part's grasp search cheaper with the same answer to the
  bit: its units on worker processes the cell starts once, and a closing line's builds in one numpy pass per
  check. Switch `batched_builds` on for a cell only once `scripts/checks/batched_builds.py` passes on its PC
  ([05](guide/05-pick-loop.md), section 2).
* `colour_check` (**on**, the owner's choice of 2026-10-08): a part whose pixels show another colour than the
  object label it was mapped onto is no target, and the VLM names the colour where the pixels cannot tell; `log`
  only says so, `off` judges nothing ([05](guide/05-pick-loop.md), 5.4). `colour_check_clipped` (**exclude**,
  the owner's choice of 2026-10-09) leaves out of the counts every pixel the camera clipped in some channels and
  not all, where an orange part's red channel clipped and read yellow; `keep` counts them, as before.
* `follow_parts` (**off**, `enabled`; 2026-10-09): a task grounds its first pick and every trigger with the
  detector and follows its parts with SAM2 alone in between, all or nothing per frame, grounding wherever anything
  is in doubt ([05](guide/05-pick-loop.md), 5.4).

**The closing direction has no `grasping` key, on purpose.** A program's `closing_axis`, the opt-in filter
that keeps only the grasps heading within 30 degrees of an axis, is its own choice, through `GraspMotion`,
`Locator.look_around` or `Scene.grasps`. The cell-level key is `robot.natural_closing_axis`
([01](guide/01-configuration.md)): it chooses the way round of every camera grasp and push, and never
filters one out ([05](guide/05-pick-loop.md), 5.2).

### 4.1 `recovery`, key by key

`recovery` is what a pick does after an attempt failed. Nothing of it runs until `enabled: true`, and then
only the actions both the mode's profile (section 2) and `allowed_actions` name.

| Key | Ships | What it does |
|---|---|---|
| `enabled` | `false` | arms the loop that runs after a failed attempt |
| `apply_modes` | `auto`, `dense_clutter` | the modes it runs in; `easy` never |
| `allowed_actions` | `()` | the actions it may plan, inside the profile: `rescan`, `next_target`, `nudge_target`, `container_agitate` |
| `max_recovery_actions` | `2` | how many actions one pick may run |
| `per_action_budget` | `()` | `(action, count)` caps; a count of `0` switches that action off |
| `critical_parts` | `false` | whether the cell's parts are critical: `false`, a boxed-in part is pushed first and the push may rearrange the scene, a blocker cleared where no push plans; `true`, nothing is pushed and a blocker is cleared. A console run can set it for itself (*Kritische Teile*) |
| `blocker_grasp_tries` | `3` | how many of a blocker's grasps the clearing tries, best first, each judged from the look before the arm leaves for it; 1 to 6 |
| `fixture` | unset | the operator's box (`center_mm`, `half_extents_mm`): required for `container_agitate`, **optional** for the push, whose push box it only narrows |
| `fixture.push_distance_mm` | `30` | how far a push moves the part when nobody asks; `30` without a fixture. Changing it or `max_nudge_mm` means declaring the box too, which then also narrows the push box |
| `fixture.max_nudge_mm` | `50` | the longest push the cell allows; `50` without a fixture |

**A refusal before anything moved falls through** to the next action for the same failure: it stays in
the trail, spends no budget and never re-runs the pick. A motion that failed once the arm moved ends the
loop. The service ends a pick the loop ran out on as `recovery_exhausted` (a pick that found nothing
keeps `no_target`), and one whose push stopped once something may have moved as
`unsafe_recovery_refused`: the campaign ends, and the service refuses every pick until a new `PickRun` or
console run starts. Clear the cell first; that run's first pick drives the arm from where it stopped.

**`rescan` on a wrist cell.** A pick handed looks has had its rescans, its looks and its one generated
view, so recovery takes `rescan` out of that pick's actions and says so at INFO. A look-less wrist pick and
a fixed camera keep their rescan where they stand.

**`next_target` skips the part that failed.** It gets an exclusion zone: its BASE XY centre, a radius of
30 mm or half its footprint diagonal, whichever is larger, for this pick and the next two, same label
only. The next attempt skips any segmentation of that label whose centre lies in a zone, beside the label
gate, and on a wrist cell looks around again for the new part. It also follows a `motion_plan_refused`,
because the new part's motions are judged in full. Where only excluded parts of the label are left, the
pick stops and says so. Before it drives a wrist pick's looks again, a toggle's count is read: one
**nobody can vouch for** ends the pick as a gripper fault before any look is driven again, and `PickRun`
and the console run stop on it.

**`nudge_target`, the push, is `dense_clutter` only, and a wrist camera's.** It runs inside a wrist
camera's pick attempt, after the looks judged the grasp: **a fixed-camera cell never pushes.** It needs
every one of these, none of which asks a person:

- the parts are not critical (`critical_parts: false`, or the run says so);
- the attempt failed `all_collided`, every grasp of the part was judged from the look and refused there,
  or its approach was blocked where `approach_validation` is on, and a neighbour stands within 25 mm of
  the part in the fused clouds of the looks; never on `no_valid_grasp` or `no_candidates_generated`;
- `nudge_target` in `allowed_actions` and a budget left: one push per part, two per pick, five per
  campaign; a spent budget means no more pushes, never a stop; and an attempt left to pick the part from
  afterwards, so `max_attempts: 1` never pushes. **No `fixture` needed**: a declared one only narrows the
  push box;
- jaws that read open: a toggle's count says open and knows it, a hand that measures its width stands
  within 2 mm of fully open; the push reads the hand and never writes it. A count that says closed is no
  push; a toggle's count **nobody can vouch for** when the push reads it, once its plan and budgets
  passed (DO0 switched at the pendant mid-pick, or unreadable), ends the pick as a gripper fault, and so
  does a hand that measures its width found **not connected** or unreadable; `PickRun` and the console run
  stop on it. A push refused before that read falls through, and `next_target` reads the count before it
  drives the looks again. The count is read again before each contact leg and once the arm is up, and one
  nobody can vouch for there stops the push where the arm stands;
- a controller that can move: one found stopped, or unreadable, ends the pick
  `controller_not_operational` with nothing commanded;
- a planned direction: the landing plus 15 mm inside an **automatic push box**, the workspace box
  intersected with the table the camera saw, shrunk by 30 mm, or a declared container's interior; the
  `fixture` box can only narrow it;
- a part at least 15 mm tall, resting on the support, and not reaching up to the palm. Where no look saw
  its foot, the table seen round it says whether it stands there.

**The push may rearrange the scene** (the owner, 2026-10-03). The part may shove a lower neighbour ahead
of it, and the fingers may brush one beside their stroke; the fingers come down 10 mm from every
neighbour on table the camera saw, the housing keeps 25 mm from every neighbour tall enough to reach it,
and whatever is shoved lands, like the part, on seen table 45 mm from its edge. Nothing is pushed over an
edge. Every neighbour the push may touch is held out of the camera world whole while it runs, and the
whole push is judged from where the arm stands before it leaves for P0.

**Critical parts** (`critical_parts: true`) are never pushed. Where every grasp of the part collided, or
was refused ahead, a neighbour in its way is gripped, set down on a free spot the camera saw or at a place
a person taught (`orchestrator.blocker_place`), and the part is picked after it. Its grasps are tried
best first, at most `blocker_grasp_tries`. The hand needs room for that: a neighbour within about 20 mm
of the part leaves the hand no room to grip it, and a part boxed in that tightly stays boxed.

**The distance.** A push moves the part `push_distance_mm`, 30 by default. `push_mm` on `PickRun` or the
API request asks for another distance up to `max_nudge_mm`; longer, or under 10 mm, is refused with a
sentence, never shortened (`422 push_distance_refused` on the console). To let a request reach 50 mm with
a 30 mm default, leave `max_nudge_mm` at its 50. Where nobody asked and 30 mm frees no direction, the
planner tries 40, then 50 mm, up to `max_nudge_mm`; a distance a person asked for is never lengthened.

**After a stop.** A down leg refused before it was sent is a fall-through, the arm back at the look
first (the lead's ruling, pending the owner); one refused with a status not known as "nothing sent"
(`unsupported`, `connection_error`, `controller_rejected`) stops, a safe stop URSim will count.

**Check `z_min` on the cell.** Every TCP point of a push stays 20 mm above the workspace's `z_min` and 20
mm inside its other faces, so the base tree's `z_min: 100` refuses every push on a table level with the
robot's base. **And give it more than one view.** A push needs table the camera saw wherever the part may
land and wherever the fingers come down. A shadow or a dip among seen table counts as ground, the edge of
what was seen never does; the fused looks of a wrist pick see past the part, and a declared container
answers it outright. A part whose lower side no view saw is pushed only where the table round it says it
stands there.

**`container_agitate`** is refused at load unless `support.container` declares its interior box (6.2).

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
3. Use `--config-subpolicies` in `auto`, or the constructor wins. See section 3.2.
4. The mode has to be one the block can fire in. See section 2.
5. Set the block's operative value, not only `enabled: true`. See section 3.1.
6. Read the right number. For a fail-closed gate, a drop in picks is the expected result, so
   counting lift is the wrong metric for most of these blocks.
7. Gate on the log line rather than the exit code. A crashed simulation process hangs instead of
   returning, so a wall-clock timeout is not a pass.

## 6. What the schema refuses

### 6.1 The declared-unwired list

`RobotGraspingConfig.UNWIRED_SWITCHES` is a list of flags whose value reaches
`EffectiveGraspingConfig`, which is the cell's own telemetry, and is then read back by nothing. It holds
one flag today, `occlusion.hard_reject_enabled`. Turning one on changes what the cell reports about
itself and not one thing about what it does. A model validator refuses to load a config that sets one,
with the reason and a pointer to this section, so an operator meets the truth where they set the value
rather than in a document they may never open.

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

### 6.2 What recovery refuses at load

Each of these is refused at load, so a cell never boots believing it can recover in a way it cannot:

- `container_agitate` in `allowed_actions` without a `fixture` (the push needs none);
- `container_agitate`, in `allowed_actions` or in `per_action_budget`, unless `support.container`
  declares `interior_min_mm` and `interior_max_mm`; a `floor_height_mm` alone does not count. A config
  with neither a fixture nor a declared container is told both in its first refusal;
- `push_distance_mm` or `max_nudge_mm` under 10 mm or over 50 mm, and a `push_distance_mm` longer than
  `max_nudge_mm`. A local tree that still writes the old `max_nudge_mm: 5`, or a `max_nudge_mm` under 30
  with no `push_distance_mm`, no longer loads, on purpose: delete the line to take 50 mm, or write 10 to
  50 mm with a `push_distance_mm` no longer than it;
- a removed action or mode, `next_viewpoint` or `dense_autonomous` among them, with the name to use
  instead.

## See also

* [`config/all_keys/`](../config/all_keys/), the generated reference tree, every key with its
  default
* [`src/config/`](../src/config/README.md), how profiles layer and how to ask the tree about itself
* [`grasping-math.md`](grasping-math.md), the formulas these blocks evaluate
* [`src/robot/grasping/`](../src/robot/grasping/README.md), the package itself
* [`src/robot/execution/autonomous_grasp/`](../src/robot/execution/autonomous_grasp/README.md), the
  composition root that turns this config into carriers
