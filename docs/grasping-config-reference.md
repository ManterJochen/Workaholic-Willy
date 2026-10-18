# The `robot.grasping` config, and why enabling a block often does nothing

[Workaholic-Willy](../README.md) | siblings: [`grasping-math.md`](grasping-math.md),
[`safety-math.md`](safety-math.md)

Read sections 2 and 3 before you enable or measure any `robot.grasping` block. The block is small since
the owner's cleanup of 2026-10-16 and 2026-10-17: the grasp mode, the generator, the record file, the two
learned blocks and the bin. The blocks that changed no pick of the owner's cell left with their code, and the
switches the cell ran are fixed at its values. What is left can still mislead: a mode decides what a pick may
do, and a block a mode or a constructor never reaches turns an honest measurement into a confident wrong answer.

This page is not a key reference. [`config/all_keys/`](../config/all_keys/) is: a complete tree in
which every key the schema accepts is written out with what it does, its default and its legal
values. It validates on its own, so it is checkable rather than merely documentation, and nothing
loads it by default:

```bash
python -m src.config --data config/all_keys                       # the reference tree validates
python -m src.config explain robot.grasping.default_mode           # type, default, tier, who set it
python -m src.config where container --tier all                   # search the schema, not the files
python -m src.config decisions                                    # only what differs from default
```

What you get here is the part a generated index cannot give you: which blocks reach the pick path at
all, and what goes wrong when you assume they do.

## 1. How the config reaches the stack

`AutonomousGraspService.from_robot_config` reads `robot.grasping`, builds an
`EffectiveGraspingConfig`, and installs a set of carriers on the orchestrator. That is the path a
real cell takes, and it is the only path that honours the config.

`from_components` is the other constructor, and it leaves `effective_config` as `None`. Every
`grasping.*` block is then silent, because nothing built it. Several simulation runners use this
constructor and set what they want in runner code, per command-line flag. So a simulation run can do what
its tree's keys say it does not, and a key changed there changes nothing. Section 3.2 is the same trap seen
from the other side.

## 2. Grasp modes, the gate in front of every block

`robot.grasping.default_mode`, or the service's `mode=` argument, selects a behaviour profile. The
profiles live in `src/robot/execution/autonomous_grasp/config.py`, in `_PROFILES`, and that table is the
authority.

| Mode | Sampling | Recovery actions allowed |
|---|---|---|
| `easy` | single object | none, and that is a guarantee rather than a default |
| `auto` | auto | `rescan` |
| `dense_clutter` | dense point cloud | `rescan`, `nudge_target` |

`config/robot/robot.yaml` ships `dense_clutter`, the owner's cell's and the schema's default since
2026-10-17. The simulation runners default to `easy`.

`next_viewpoint` left both profiles on 2026-09-29, merged into `rescan`: with the viewpoint planners
gone nothing moved the camera for it, so it re-perceived exactly as `rescan` does. `next_target`, the rescan
that skipped the part a pick failed on, left both on 2026-10-16 with the `recovery` block (the owner's
cleanup). A run that still names either is refused, `removed on purpose: use rescan`.

`closed_loop` and `dense_autonomous`, the two modes that refined and verified, were removed on
2026-09-29 with the two-scan pre-grasp refinement they ran; no shipped config switched it on, and it
never ran on a physical arm. A tree, a preset or a call that still names one is refused with the mode
to name instead: `auto` for `closed_loop`, `dense_clutter` for `dense_autonomous`. `nudge_target`, which
only `dense_autonomous` allowed, moved to `dense_clutter`.

**The mode's profile is an outer gate** that no run can widen: `auto` allows `rescan`, `dense_clutter`
adds `nudge_target`. `rescan` re-runs the pick, which perceives and ranks afresh. `nudge_target` is **the
push**, and it runs inside a wrist camera's pick attempt only: the service's recovery loop never plans it.
Section 4.1 is the recovery every cell runs.

**This is why a measurement can be worth nothing.** The mode-gated blocks, `feasibility`, `occlusion`,
`ordering`, `uncertainty`, `success_model` and `approach_validation`, each with its own `apply_modes`, left on
2026-10-16 with their code: switched on one at a time, none changed a pick of the owner's cell's replay bank
(479 picks). One mode gate is left, the recovery's: `build_effective_config` runs the cell's recovery in
`auto` and `dense_clutter` alone, so a measurement in `easy` measures no recovery.

Being built is not the same as being acted on. `decision` was built in every mode and fired only outside
`easy`, `approach_validation` installed its carrier in every mode and fired only in `dense_clutter`; both
left on 2026-10-16.

`verification` was the third until 2026-09-29, when it was removed: no attempt consulted the verifier
it built. The hold an attempt reports is the execution policy's own check after its close, which reads
the gripper's `is_object_detected` and `hold_evidence` (a Robotiq's gOBJ). A tree that still writes the
block is refused at load with the sentence that says so.

The recovery runs in `auto` and `dense_clutter`, and its one motion, the push, in `dense_clutter` alone,
inside a wrist camera's pick attempt (4.1).

That last shape, a carrier that is set and never read, is the one the wiring guard cannot see. The
guard's claim is that a flag reaches a runtime carrier, and `EffectiveGraspingConfig` is a runtime
carrier. Reaching a carrier and the pick path acting on it are different claims, and knowing that
limit is part of using the guard.

## 3. The three traps

Each one turns a measurement into a lie that reads as good news.

### 3.1 The zero-weight trap

Three blocks shipped their operative value at zero, `feasibility.weight`, `ordering.unlock_weight` and
`uncertainty.ranking_penalty_weight`, so `enabled: true` and nothing else computed the signal and then
multiplied it by nothing. All three left on 2026-10-16 with their blocks. The one block left that changes no
grasp when switched on is `deep_ranker`, a shadow by design (4).

### 3.2 The constructor wins

`build_decision_layer` preferred a constructor argument over the config block, so a simulation runner's
hand-built decision engine ran whatever `grasping.decision.enabled` said; the layer and the runner's
`--config-subpolicies` left on 2026-10-16. The trap stays in its other form: a runner that builds through
`from_components` (section 1) sets in code what a tree says in YAML. Run `--boot config` where the tree is
what you measure.

### 3.3 Two things are called "mode"

On `run_multiview_pick`, `--mode` selects the camera set: `eth1`, `eth2`, `eth3` or `sides`. The
grasp mode is `--grasp-mode`. They are unrelated, and the logs call both of them mode.

A fourth trap left on 2026-09-29: two blocks were called "recovery". `grasping.dense_recovery`, the
strategy half, built a policy and a strategy no pick consulted, and it was removed; a tree that still
writes it is refused at load. The recovery a pick runs is no block since 2026-10-16 (4.1).

## 4. What each block is for

Keys and defaults are in [`config/all_keys/`](../config/all_keys/). This is the one-line purpose and
the mode gate. Every block in this table ships `enabled: false`.

| Block | What it does | Fires in |
|---|---|---|
| `deep_ranker` | the learned ranker in shadow: it scores, the telemetry records, the order never changes | every mode |

The `decision`, `feasibility`, `occlusion`, `ordering`, `uncertainty`, `success_model`, `performance`,
`approach_validation` and `watchdog` blocks left on 2026-10-16 with their code, and a tree that still writes
one is refused at load with the sentence that says why. `ordering` had a second precondition that no key
expressed: it could only act where the loop was free to choose which object goes first, and with a target
label, which every task names, the choice is made before ordering was consulted.

The always-on tunings are a separate family and are not gated by mode:

* `calculator` picks the generator, `geometric` or `deep`. `deep` with no readable artifact refuses
  to build the cell rather than falling back, and no trained weights ship here. A cell also refuses
  trained weights whose proof has not passed, which today is all of them (2026-10-09).
* `deep_generator` sizes and points that learned generator; it is inert while `calculator` is
  `geometric`.
* there is no `geometry` block: the part's body over its support is the only stage that proposes grasps, and
  every cell cuts a 2 mm rim off every mask for the cloud it builds the footprint from, and for nothing else
  (the D415 smears a part's far edge into a ramp the footprint follows), every mask pixel the stage does not
  read placed in the part's visual hull: on 23 recorded cubes the grasp centre went from 2.53 to 0.58 mm off at
  the median, and stayed at 0.59 with 35 % of the depth knocked out ([05](guide/05-pick-loop.md), section 2).
  The silhouette stage, its depth levers and `inflate_mm` left with the block on 2026-10-16, and a tree that
  still writes it, or `isotropic_radial_closing`, is refused at load.
* `support` says where the world's floor is, and the bin or tray is `support.container`, not
  `grasping.container`.
* there is no `fusion` block: a cell fuses its fixed cameras by itself wherever `camera.cameras.rigs`
  declares two or more enabled RGB-D rigs with their calibration, the primary among them
  ([multiview/](../src/robot/grasping/multiview/README.md)), and a tree that still writes the block is
  refused at load (2026-10-16).
* The collision envelope the calculator checks is the named hand's registry file (`config/grippers/<model>.yaml`,
  the 2F-85's where no hand is named); `gripper_geometry`, which restated it, is refused at load (2026-10-17).
* `default_mode` and `record_log_path` sit at the top of the block: the mode is `dense_clutter` unless set,
  and a record path left null is `grasp_records.jsonl` in the log folder, so a cell keeps its records whether
  or not its tree names a file (both since 2026-10-17). A pick makes up to five attempts, the cell's
  (`MAX_ATTEMPTS`); `max_attempts` left the block on 2026-10-17, and a service built in code may still ask
  fewer (`from_robot_config(max_attempts=...)`).
* Twelve switches left the block on 2026-10-17 at the values the owner's cell ran, and a tree that still
  writes one is refused at load. The ones below say what each now always does.
* The parts beside the one grasped are obstacles (`scene_obstacles` until then): the calculator holds every
  object the camera saw beside the part as an obstacle, whether a prompt named it or not, with what the part
  stands on taken out; a part whose every grasp meets one is `all_collided`, the failure a blocker or a push
  answers ([05](guide/05-pick-loop.md), section 2).
* Tilted approaches join the vertical ones (`side_approaches` until then, the owner's decision of 2026-10-01):
  the search offers every tilt the hand fits at, scored by the room each grasp keeps, vertical on a tie, and a
  grasp 15 degrees or more off vertical only through space a depth ray saw.
* A Robotiq hand grips a part from inside its bore where no outer grasp holds (`inner_grasps` until then, the
  owner's decision of 2026-10-14: "Nur wo außen nichts geht"): it comes down narrow into a bore the camera saw
  free, opens at the close's force onto the wall and closes to let the part go. Every such grasp comes after
  every outer grasp, so it is taken where no outer grasp is admitted. No other hand is offered one
  ([generation README](../src/robot/grasping/generation/README.md)).
* The two that come with them are every cell's since 2026-10-17 (their keys under
  `robot.safety.planning_world.perceived` left that day): the supports the parts stand on are found and held as
  solid, 2 mm over what they take out (`perceived.SUPPORT_ALLOWANCE_MM`, [04](guide/04-robot-and-safety.md), 5.5). A
  library caller building `WorldBuildTuning` by hand gets neither, as before.
* The first good part is taken (`first_good_part` and `good_part_score` until then, the owner's "speed first" of
  2026-10-08): a look's parts are computed in the order they stand apart and the first whose grasp scores 0.6 is
  taken, the best where none is good ([05](guide/05-pick-loop.md), section 2).
  `fine_pass_waits`, `workers` and `batched_builds` left on 2026-10-16 with the support-footprint stage's own
  search, which no pick asked since 2026-10-14; a tree that still writes one is refused at load.
* A part of another colour is no target (`colour_check` until then, the owner's choice of 2026-10-08): a part
  whose pixels show another colour than the object label it was mapped onto is no target, and the VLM names the
  colour where the pixels cannot tell ([05](guide/05-pick-loop.md), 5.4). Every pixel the camera clipped in some
  channels and not all is left out of the counts (`colour_check_clipped: exclude` until then, the owner's choice
  of 2026-10-09), where an orange part's red channel clipped and read yellow.
* The parts beside the named ones are named, read off the depth (`name_neighbours` until then); a weak view of
  the part sends a wrist pick on to its next look (`weak_look_trigger`); and a task's own bin and drops are
  painted out of the image its detector reads (`hide_own_places`) ([05](guide/05-pick-loop.md), 5.1 and 5.4).
* A task follows its parts from pick to pick (`follow_parts` until then, 2026-10-09): it grounds its first pick
  and every trigger with the detector and follows its parts with SAM2 alone in between, within 10 mm a pick and
  20 mm since its grounding, all or nothing per frame, grounding wherever anything is in doubt
  ([05](guide/05-pick-loop.md), 5.4).

**The closing direction has no `grasping` key, on purpose.** A program's `closing_axis`, the opt-in filter
that keeps only the grasps heading within 30 degrees of an axis, is its own choice, through `GraspMotion`,
`Locator.look_around` or `Scene.grasps`. The cell-level key is `robot.motion.natural_closing_axis`
([01](guide/01-configuration.md)): it chooses the way round of every camera grasp and push, and never
filters one out ([05](guide/05-pick-loop.md), 5.2).

### 4.1 The recovery every cell runs

There is no `recovery` block since 2026-10-16 (the owner's cleanup): every cell built from config runs the
recovery the owner's cell ran (`CELL_RECOVERY` in `src/robot/execution/autonomous_grasp/config.py`), and a
tree that still writes the block is refused at load with the sentence that says to delete it.

| What | The cell's value |
|---|---|
| the modes it runs in | `auto` and `dense_clutter`; `easy` never |
| the actions, inside each mode's profile | `rescan`, and `nudge_target`, the push |
| how many actions one pick may run | 2 |
| how often a pick rescans | once |
| how many of a blocker's grasps the clearing tries | 3, best first, each judged from the look before the arm leaves for it |
| where a cleared blocker goes | where the parts go; a run may set it aside instead |
| how far a push moves the part | 30 mm unless a run asks, 50 mm at most |
| whether the parts are critical | `runtime.task.critical_parts` in `app/runtime.yaml`, `false` unless set; a console run can set it for itself (*Kritische Teile*) |

The service ends a pick the loop ran out on as `recovery_exhausted` (a pick that found nothing keeps
`no_target`), and one whose push stopped once something may have moved as `unsafe_recovery_refused`: the
campaign ends, and the service refuses every pick until a new `PickRun` or console run starts. Clear the
cell first; that run's first pick drives the arm from where it stopped.

**`rescan` on a wrist cell.** A pick handed looks has had its rescans, its looks and its one generated
view, so recovery takes `rescan` out of that pick's actions and says so at INFO. A look-less wrist pick and
a fixed camera keep their rescan where they stand.

**`nudge_target`, the push, is `dense_clutter` only, and a wrist camera's.** It runs inside a wrist
camera's pick attempt, after the looks judged the grasp: **a fixed-camera cell never pushes.** It needs
every one of these, none of which asks a person:

- the parts are not critical (`critical_parts: false`, or the run says so);
- the attempt failed `all_collided`, or every grasp of the part was judged from the look and refused
  there, and a neighbour stands within 25 mm of the part in the fused clouds of the looks; never on
  `no_valid_grasp` or `no_candidates_generated`;
- a budget left: one push per part, two per pick, five per campaign; a spent budget means no more pushes,
  never a stop; and an attempt left to pick the part from afterwards, so a service built with
  `max_attempts=1` never pushes.
  **No fixture needed**: the automatic push box bounds it;
- jaws that read open: a toggle's count says open and knows it, a hand that measures its width stands
  within 2 mm of fully open; the push reads the hand and never writes it. A count that says closed is no
  push; a toggle's count **nobody can vouch for** when the push reads it, once its plan and budgets
  passed (DO0 switched at the pendant mid-pick, or unreadable), ends the pick as a gripper fault, and so
  does a hand that measures its width found **not connected** or unreadable; `PickRun` and the console run
  stop on it. A push refused before that read lets the attempt go on. The count is read again before each
  contact leg and once the arm is up, and one nobody can vouch for there stops the push where the arm
  stands;
- a controller that can move: one found stopped, or unreadable, ends the pick
  `controller_not_operational` with nothing commanded;
- a planned direction: the landing plus 15 mm inside an **automatic push box**, the workspace box
  intersected with the table the camera saw, shrunk by 30 mm, or a declared container's interior;
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
best first, at most three. The hand needs room for that: a neighbour within about 20 mm
of the part leaves the hand no room to grip it, and a part boxed in that tightly stays boxed.

**The distance.** A push moves the part 30 mm. `push_mm` on `PickRun` or the API request asks for
another distance, up to 50 mm; longer, or under 10 mm, is refused with a sentence, never shortened
(`422 push_distance_refused` on the console). Where nobody asked and 30 mm frees no direction, the planner
tries 40, then 50 mm; a distance a person asked for is never lengthened.

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

## 5. How to measure a block

One block per run, each against a baseline in the same mode with the same flags:

```bash
# baseline for this mode
python -m src.willy_sim.run_multiview_pick --runs 5 --boot config \
    --scene tall --rendered-depth --grasp-mode auto

# the same thing with exactly one block layered on, from a profile layer of your own
python -m src.willy_sim.run_multiview_pick --runs 5 --boot config \
    --scene tall --rendered-depth --grasp-mode auto \
    --profile <your layer>
```

`--profile decision` stacked `config/robot/robot.decision.yaml` on top of the loaded tree, which
turned the gate on and nothing else, so a measured difference was attributable to it; the profile left
on 2026-10-16 with the decision layer, and a profile of your own that switches one block does the same.
Five rules:

1. Use `--boot config`, or the measurement is about the runner and not about the cell. See section 3.2.
2. Use `--rendered-depth`, or the measurement is about a ground-truth oracle that reports the
   object's centre rather than a surface any real depth camera would deliver.
3. The mode has to be one the change can act in: the recovery acts in `auto` and `dense_clutter` alone.
   See section 2.
4. Read the right number. A shadow such as `deep_ranker` changes no pick by design, so counting lift
   measures nothing; read the telemetry it writes.
5. Gate on the log line rather than the exit code. A crashed simulation process hangs instead of
   returning, so a wall-clock timeout is not a pass.

## 6. What the schema refuses

### 6.1 The declared-unwired list

`RobotGraspingConfig.UNWIRED_SWITCHES` was a list of flags whose value reached
`EffectiveGraspingConfig`, which is the cell's own telemetry, and was then read back by nothing: turning one
on changed what the cell reported about itself and not one thing about what it did, so a model validator
refused a config that set one. Its last flag, `occlusion.hard_reject_enabled`, left on 2026-10-16 with its
block, and the list went with it.

The rule it kept stays: every flag left must move the runtime, which
[`tests/robot/grasping/types/test_grasping_wiring_guard.py`](../tests/robot/grasping/types/test_grasping_wiring_guard.py) checks flag by flag, so a
flag may not be quietly acceptable and inert. A block or a switch removed on purpose is refused at load with
the sentence that says what replaced it ([`src/config/schema/_removed.py`](../src/config/schema/_removed.py)),
and `python -m src.config migrate PATH --write` rewrites an old tree to the keys this Willy reads.

`feasibility` carried a second warning that was not about wiring: a good overhead top-down grasp is itself
near singular, meaning a high Jacobian condition number, so its `ik_quality` sub-signal demoted exactly the
grasp that was wanted, and a measured UR5e overhead pick went from 10 of 10 to 0 of 10. It left on 2026-10-16
with its block.

### 6.2 What recovery refuses

- a tree that still writes `robot.grasping.recovery`, at load: delete the block, as its sentence says; the
  one choice that stayed with the cell is `runtime.task.critical_parts`;
- a run that names a removed action (`next_target`, `container_agitate`, `next_viewpoint`), or a tree or a
  call that names a removed mode (`dense_autonomous`, `closed_loop`), with the name to use instead.

## See also

* [`config/all_keys/`](../config/all_keys/), the generated reference tree, every key with its
  default
* [`src/config/`](../src/config/README.md), how profiles layer and how to ask the tree about itself
* [`grasping-math.md`](grasping-math.md), the formulas these blocks evaluate
* [`src/robot/grasping/`](../src/robot/grasping/README.md), the package itself
* [`src/robot/execution/autonomous_grasp/`](../src/robot/execution/autonomous_grasp/README.md), the
  composition root that turns this config into carriers
