# The pick service (`src/robot/execution/autonomous_grasp`)

`AutonomousGraspService` runs one pick attempt per `pick()` and answers with one frozen
`AutonomousGraspReport`: perceive, generate and rank grasps, gate, move and log, plus whichever
opt-in layers the grasp mode and the config switch on. You reach it through `Cell.build()` and
`PickRun`; build it yourself only when you hold a grasp calculator and a perception source of your own.

```python
from willy import Cell, load_tree

cell = Cell.rehearsal(load_tree("console_dummy").robot)   # a dummy arm and a synthetic scene
service = cell.build()                                     # the AutonomousGraspService behind the cell
with cell.connected():
    report = service.pick()                                # one attempt; a refusal is an outcome
print(report)                                              # outcome, candidates, camera world, layers
```

At a real cell `Cell.from_tree(load_tree(), prompt="a red cube")` builds the same service with the
cell's cameras, models and planner, and
[12_pick_with_the_camera.py](../../../../examples/real_robot/12_pick_with_the_camera.py) runs a campaign of them.

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `AutonomousGraspService` | `Cell.build()`, `build_real_cell(robot_cfg, prompt=)`, `from_robot_config`, `from_components` | `pick(mode=None, look=..., both_faces=False)`, `put_back(report)` | `AutonomousGraspReport`, `HandlingReport` |
| `GraspMode` | `resolve_grasp_mode(value)`, or `Cell(..., mode=)` at the build | | `easy`, `auto`, `dense_clutter` |
| `PickPrompt` | `PickPrompt.from_text(text)` | `service.set_prompt(text)` | the prompt it replaced |

`resolve_grasp_mode` also takes the aliases `single`, `single_object` and `dense`, and `None` is
`auto`. `closed_loop` and `dense_autonomous` (with their aliases `closedloop` and `autonomous`) were
removed on 2026-09-29 with the two-scan pre-grasp refinement they ran, and are refused with the mode to
name instead: `auto` for `closed_loop`, `dense_clutter` for `dense_autonomous`, which took over the
`nudge_target` recovery. The profiles bound what recovery may ever do: `easy` nothing, `auto` `rescan`
and `next_target`, `dense_clutter` those two and `nudge_target`, the push that runs inside a wrist camera's
pick attempt ([recovery/](../../grasping/recovery/README.md)). `mode=` on `pick()` changes the behaviour profile of one attempt,
never the sampler the service was built with -- so the mode a cell RUNS IN is chosen at the build,
`Cell.from_tree(tree, mode="dense_clutter")`, and asking a service built in one sampler for another
comes back `MODE_NOT_AVAILABLE`.
[`simulation/06`](../../../../examples/simulation/06_grasp_modes_and_what_each_needs.py) prints
every mode, what each one locks and which of them this cell can run; `willy` exports the service,
the report, the outcome, `GraspMode` and `PickPrompt`. A loop of your own calls `service.pick()` in
it, inside `with cell.connected():`, and reads `report.outcome` and `report.layers_that_ran()` off
each attempt.

`pick(look=...)` takes a `JointPositions`, `"home"`, or a list of them ([looks.py](../looks.py)).
`pick(multi_view=False)` looks from the first look only and generates no view, for one pick. A task
([`task.py`](../task.py)) sets its closing axis on the service for its length with `set_closing_axis(axis)`,
which answers the axis it replaced; `closing_axis_refusal(axis)` says why one would be refused (a name that is no
axis, or a cell whose motion aligns every grasp to base X), with nothing changed. `found_nothing(report)`,
`only_excluded(report)` and `only_kept_out(report)` say how a pick that took nothing ended, and
`detector_failures()` counts the detector failures of the cell's grounding cameras, so a task tells "nothing
there" from "the model could not look".
**A wrist camera** hands its looks to the pick loop (`orch.looks`, `orch.both_faces`, taken back in a
`finally`, so the next pick inherits neither): it looks from each, fuses each with the ones before,
stops at the first valid grasp with no rescan reason and may generate one more view as the last resort
([grasping/loop](../../grasping/loop/README.md)). A look refused before anything was sent is skipped and
said; a pick that reaches none of its looks, or whose look motion may have moved the arm, ends
`EXECUTION_FAILED` (`CANCELLED` on a stopped controller) with nothing perceived. **A fixed camera**
moves to each look in turn and perceives there until one finds something. Nothing is said to the hand
before a look. Without `look=` the pick perceives from where the arm stands: a wrist pick then rescans
there, and a fixed camera does not move. `PickRun` and the console hand every pick the cell profile's
looks (`service.configured_looks`, from `robot.look_joint_positions_deg`), else `"home"` on a wrist
camera. The decision path decides on the wrist camera's `look_around()` and hands that judgement on with
`go_on_with`, so the pick does not look twice.

`pick(both_faces=True)`, off by default, asks that both jaw contact faces of the chosen grasp be seen
before gripping: the owner's switch for safety-critical processes. A wrist pick looks on for them, its
one generated view included; a fixed camera grips only a grasp its cameras showed both faces of. Where
no view showed both, nothing is gripped: `NO_VALID_GRASP`, the unseen face named in its failure line. A
decided `GRASP_NOW` refused so keeps the engine's decision on the report. A suction cup has no faces to
see. With it, the way round comes from the cell's `robot.natural_closing_axis`, which turns every grasp the
nearer way round, and one axis to close along from `GraspMotion(closing_axis=...)`, which keeps only the
grasps within 30 degrees of it ([grasping/motion](../../grasping/motion/README.md)): both act before the
grasps are judged. The simulator's twist, `GraspMotion(align_closing_to_base_x=True)`, turns every grasp
after the judging, so beside `both_faces` it raises `ValueError` before the controller or the hand is
asked.

`pick(follow=kept, follow_looks=True)` is how a task follows its parts (`robot.grasping.follow_parts`, the owner,
2026-10-09; [`task.py`](../task.py) hands both, off by default): `follow` is the parts its last pick kept
(`src/robot/perception/kept_scene.KeptScene`), which the pick's first look finds again with SAM2 on their boxes and no
detector, grounding the frame as before where any check fails; `follow_looks` has every later look find the first
look's parts by their projected boxes and rank the part it keeps first. Both are a wrist camera's, handed to the pick
loop (`orch.follow`, `orch.follow_looks`) only where asked and taken back in the same `finally`; a fixed camera ignores
them. The first look takes `follow`, so a pick the recovery loop runs again grounds.

`pick(every_look=True)` is the console's "Alle Posen" (the owner, 2026-10-08 night): a wrist pick visits every look it
was handed, the early stop off, and goes on with the last look's grasp, fused over every look. Handed to the pick loop
(`orch.every_look`) only where asked and taken back in the same `finally`; ignored with `multi_view=False` and on a
fixed camera. `from_robot_config` sets two of the cell's own switches on the pick loop where its tree turns them on,
both off by default: `robot.grasping.weak_look_trigger` (`orch.weak_look`: a weak view of the part sends a wrist pick
on to its next look) and `robot.grasping.hide_own_places` (`orch.hide_own_places`: the regions a task keeps out painted
out of the image its detector reads); see [grasping/loop](../../grasping/loop/README.md).

`RuntimePickService.from_robot_config` hands the pick loop the cell's `robot.natural_closing_axis`
(`BinPickingOrchestrator.natural_closing_axis`), so every pick of the service, `Cell` and `PickRun` turns
its grasps and its pushes the natural way round with no argument of its own. A motion's `closing_axis`
reaches the loop through `GraspMotion` and the policy, and a pick none of whose grasps closes along it
ends `NO_VALID_GRASP`, its failure line naming the axis and the 30 degrees, also where the frame failed
on another part (`PickAttempt.withheld`).

With recovery armed (`grasping.recovery`, as the `dense_clutter` preset arms it), a wrist pick handed
looks gets no recovery rescan: its looks and its one generated view were its rescans, so a rescan never
repeats them, and an INFO line says so. Only `next_target` runs them again within the same `pick()`, for
another part of the label: the look sequence, its early stop and at most one generated view. `PickRun`
and the console hand a wrist cell at least `"home"`, so their wrist picks count as handed looks. A wrist
pick handed no look and a fixed camera rescan where they stand. Before `next_target` drives the looks
again, a hand that toggles with no sensor is read (`why_toggle_count_unknown`, nothing asked or sent): a
count nobody can vouch for ends the pick there, `EXECUTION_FAILED` with `gripper_fault`,
`telemetry['stage']` `before_the_re_pick` and `re_pick_refused` `gripper_fault` (the trail's last step never
picked), before any look is driven again (the owner, 2026-09-30). A re-pick that drives no look reads
nothing more: its own check before the arm moves asks where the jaws stand, as at every pick start.

The report names the looks, where the object's seen surface is centred in BASE (`object_centre_mm`) and
the pose the tool closed at (`grasp_pose`), both read off the pick report. `put_back(report)` places a
lifted part back at that pose through `Robot.place`. What a wrist pick's looks came to is on the report
and readable on `service.looked_around` (`None` after a pick that ended before it looked):

| Report field | What it says |
| --- | --- |
| `looks` | the looks perceived from; a fixed camera's, the looks it moved to, the last possibly not reached |
| `looks_fused` | the looks whose views make up the grasp's cloud, the one it was ranked on first |
| `jaw_faces_seen` | (jaw 1, jaw 2): whether each contact face of the chosen grasp was seen |
| `hand_eye_gap_mm` | the median distance at which the looks measure the part's shared surface; above 6 mm a WARNING |
| `both_faces` | whether the pick asked for both faces |
| `generated_view_deg` | how far the one generated view turned about the part |
| `fused_views` | the views fused into the part's cloud: `rig@look` for a wrist look, the rig id for a fixed camera |
| `telemetry['refused_look']`, `['look_refused']` | the look, generated view or move back whose motion ended the pick, and why |
| `telemetry['looks_skipped']` | looks refused before anything was sent, each `"look: why"` |
| `telemetry['move_back_refused']` | why the arm did not go back to the look that ranked the grasp; the approach started where it stood |
| `telemetry['targets_by_look']` | how many targets each look perceived from saw, in their order: the parts past the label gate, no surface, none kept out |
| `telemetry['followed']`, `['follow_why']` | on a pick handed parts to follow: whether its first look followed them, and what it came to or why the detector grounded it |
| `telemetry['looks_followed']` | the looks whose parts were found with no detector, the first and the later looks' projected boxes |
| `telemetry['every_look']` | on a pick that drove every look ("Alle Posen"): `True` |
| `telemetry['looks_weak']` | the looks the cell's trigger judged weak, each `"look: why"` (`robot.grasping.weak_look_trigger`) |
| `pick_report.target_label` | the label of the segmentation the pick went for, its executed grasp's or its last chosen one's; `""` where it chose none |
| `pick_report.unclaimed_labels` | on a sort, the labels the label gate turned away at the pick's first look, one per part no rule claims; a surface and a part in a region the task keeps out left out |

`failure_summary()` adds the look not reached, the looks perceived from and the looks skipped, and names
the unseen face only when `both_faces` was asked and the outcome is `NO_VALID_GRASP` (on a fixed camera,
in the pick loop's own words). `render()` prints them on their own lines.

`set_prompt` changes what the next picks look for (the phrase every camera grounds, the labels the
detector's words map onto, and the label filter) with no camera reopened and no model reloaded:

```python
previous = service.set_prompt("the red cube")
report = service.pick()
service.set_prompt(previous)                       # all three back
```

A sort names several kinds at once (the owner, 2026-10-09: "Grüne Teile in die gelbe Kiste, rote in die blaue"):
the phrase is a class list, every kind grounded in one call, and the label filter takes any of the kinds
(`PickPrompt.target_labels`, with `target_label` `None`). The prompt `set_prompt` returns carries the kinds it
replaced, so passing it back puts them back too. Each pick says which kind it went for, and which parts its first
look saw that no rule claims (the detector's `ambiguous`, a word no rule names), both on the pick report:

```python
from src.models.vlm.qwen import class_list_prompt

kinds = ("green part", "red part")
previous = service.set_prompt(PickPrompt(
    phrase=class_list_prompt(["each separate green part", "each separate red part"]),
    object_labels=kinds, target_labels=kinds))
report = service.pick()
report.pick_report.target_label                    # "green part": the yellow bin's rule
report.pick_report.unclaimed_labels                # ("ambiguous",): it stays where it lies
service.set_prompt(previous)
```

To build the service from parts of your own, `calculator` and `perception` are required and have no
default, because a forgotten argument must not open a camera. A real vendor also needs the CAMERA to
BASE of its camera, from the camera section or as `frame_resolver=`:

```python
from willy import load_tree
from src.robot.execution.autonomous_grasp import AutonomousGraspService

tree = load_tree()
service = AutonomousGraspService.from_robot_config(
    tree.robot, calculator=my_calculator, perception=my_source, camera=tree.app_config.camera)
```

`build_real_cell(robot_cfg, prompt=...)` supplies all three from the tree, and is what `Cell.build()`
calls. `service.enable_record_logging(path)` appends one `GraspAttemptRecord` per pick.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `MODE_NOT_AVAILABLE` outcome | a `mode=` of another sampler than the one the service was built with | build the service in that mode |
| `ValueError`, `removed on purpose` | a mode that left on 2026-09-29: `closed_loop`, `dense_autonomous` | name the mode the sentence names |
| `EXECUTION_FAILED` with `fault` | a `RobotError`, `RuntimeError` or `OSError` during the pick | `PickRun` and the console stop the campaign on it |
| `MISSING_CAMERA_FRAME` outcome | a grasp won and no frame resolver maps it to BASE | declare the camera's calibration on its rig |
| `ValueError` | no `mode` and no `robot.grasping` block declared | declare the block, or pass `mode=` |
| `ValueError` | a real vendor with no CAMERA to BASE, or `container_agitate` with no envelope (a tree is refused at load without `recovery.fixture`; a hand-built `SceneRecoveryPolicy` raises); the push needs none | as the message says |
| `ValueError`, `TypeError` | a `policy=` on another arm or hand; both `motion=` and `policy=` | pass `motion=GraspMotion(...)` alone |
| `ValueError`, `TypeError`, before anything moves | a look list that names nothing, or an entry that is not a look; `both_faces=True` with `GraspMotion(align_closing_to_base_x=True)` | write looks as `JointPositions.deg(...)` or `"home"`; ask for one switch or the other; to close along one axis with `both_faces`, name it with `closing_axis` |
| `NO_VALID_GRASP`, the axis named | `GraspMotion(closing_axis=...)` and no grasp of the part closes within 30 degrees of it | name an axis the part's grasps close along, or none |
| `NO_VALID_GRASP`, the face named | `both_faces=True` and no view showed both jaw contact faces of the chosen grasp | give the pick a look that faces the missing side, or leave the switch off |
| `EXECUTION_FAILED`, `telemetry['look_refused']` | a pick reached none of its looks, or a look motion may have moved the arm | read the motion status; teach looks the planner admits |
| `CANCELLED`, `controller_stopped` | the controller cannot move, or cannot be read, before the pick or before a push | a person clears the stop where the arm is visible; `PickRun` and the console stop on it |
| `EXECUTION_FAILED`, `gripper_fault` | a toggle hand would not start the pick (jaws it believes closed, nobody to ask), the gripper raised, a push found nobody can vouch for the hand (a toggle's count, a width-measuring gripper not connected or unreadable), or `next_target` found the toggle's count so before it drove the looks again | look at the hand; `PickRun` and the console stop on it |
| `UNSAFE_RECOVERY_REFUSED` (`CANCELLED` where the controller stopped), `needs_person` | a push stopped once something may have moved (a toggle's count nobody can vouch for before a contact leg or once the arm is up among it), and every later pick until a person decides | clear the cell, then start a new run, or call `acknowledge_needs_person()` |
| `ValueError` from `start_campaign`, `push_distance` | a `push_mm` above the cell's `max_nudge_mm` or 50 mm, or under 10 mm | ask within the ceiling |

A programmer's error still raises from `pick()`, and so do `NotImplementedError` and `RecursionError`.

With recovery armed, the service holds **one campaign** per `PickRun` or console run
(`start_campaign(push_mm=)`, a `PushCampaign`: the push budgets, the parts `next_target` skips and the push
distance, `push_distance()`); a pick no caller started one for starts it. It hands each pick the gate that
lets a push run (`push_permitted` and the cell's `push_cell`; only a wrist camera's pick reads it), and
sets the outcome a finished recovery loop stands for: `RECOVERY_EXHAUSTED` where an action ran and none
was left, `UNSAFE_RECOVERY_REFUSED` where a recovery motion stopped once it had started; a pick that found
nothing keeps `NO_TARGET`. A push that stopped once something may have moved makes the report
**`needs_person`**: `PickRun` and the console stop, and the service **refuses every later pick**
(`UNSAFE_RECOVERY_REFUSED`, `telemetry['stopped_where_the_arm_stands']`; nothing asked, perceived or
moved, no record) until `start_campaign()`, which every new `PickRun` and console run calls, or
`acknowledge_needs_person()`. Clear the cell first: the next pick drives the arm to its first look from
wherever the push left it. A hand nobody can vouch for when the push reads it, a toggle's count or a
width-measuring gripper not connected or unreadable, ends the pick `EXECUTION_FAILED` with `gripper_fault`,
nothing pushed; a toggle's count nobody can vouch for before a contact leg or once the arm is up stops the
push where the arm stands (`needs_person`). Every push the pick considered is in `telemetry['pushes']`, a
stopped one's sentence in `telemetry['push_stopped']`, and every push whose arm left the look is a
`nudge_target` row of `recovery_actions` ([recovery/](../../grasping/recovery/README.md)).

## Status

| Capability | Evidence |
| --- | --- |
| The service on a UR5e with a 2F-85 in Isaac Sim | measured in simulation ([willy_sim](../../../willy_sim/README.md)) |
| The service on a physical arm | run on a physical cell: camera picks through `PickRun` (example 12) on a UR10 (CB3) with a wrist D415 and a `jaw_io` Hand-E; the console has not run there, nor the looks, the generated view or the push |
| A wrist pick's looks through `PickRun` and the console | pinned end to end with the repository's doubles (`tests/test_a_wrist_pick_runs_end_to_end.py`); never touched hardware |

The default attempt is open-loop. Every advanced `robot.grasping` block ships `enabled: false`
(`decision`, `uncertainty`, `feasibility`, `ordering`, `recovery`, `fusion`, `success_model`,
`deep_ranker`, `approach_validation`, `performance`), and `robot.rl.mode` ships `hybrid_ml`, which
builds no shadow router. The report's `layers` line names what actually ran, read off the attempt
rather than the config. The hold an attempt reports is the execution policy's own check after its
close (`hold_measured`), which reads the gripper's `is_object_detected` and `hold_evidence`. The
separate post-grasp verification stage, its `verification` block and the `dense_recovery` block, with
the service slots they filled (`verifier`, `verification_policy`, `recovery_policy`,
`recovery_strategy`), were removed on 2026-09-29: no attempt consulted them.

Two ways to build the service are not equivalent. `from_robot_config` reads the whole tree and fills
`effective_config`. `from_components`, for a caller holding live handles the config cannot describe,
leaves it `None`, which silences uncertainty fusion, the watchdog, breach events and the recovery
loop until the caller calls `build_effective_config` and `apply_orchestrator_overlays`.
`from_robot_config` takes live `arm` and `gripper` handles for exactly that reason.

Record logging is off unless `grasping.record_log_path` is set (it takes `${WILLY_RECORD_LOG:-}`, and
an empty value is off). Each pick carries a unique `attempt_id`. A cell that logs records while
`robot.rl.mode` is not `rl_shadow` warns once at boot, because a pairwise ranker cannot train on them.
A record's `extra` carries the looks where the pick had them: `looks_visited` on any camera handed looks
(a fixed camera's look the arm did not reach left out); `looks_fused`, `jaw_faces_seen`,
`hand_eye_gap_mm` and `generated_view_deg`, a wrist camera's alone; and `both_faces`, written only when
it was asked.
A real cell (`build_real_cell`) writes its records in the background (`write_records_in_the_background`):
`pick()` returns at once, the process's one writer writes the records in the order of their picks, each
stamped when its pick ended, a write that fails is logged and stops nothing, and whatever still waits is
written before the process ends. A cell built by hand writes each record before `pick()` returns, as
before. `build_real_cell` also starts the cell's SFE workers (`robot.grasping.workers`) and hands them to
the pick loop.
`EffectiveGraspingConfig.to_dict()` is the flat telemetry contract of 77 keys, each the state that
acted on the attempt: a block outside its `apply_modes` reads false even where the YAML says true.
`closed_loop_enabled` left it on 2026-09-29 with the refinement it flagged, and
`verification_enabled`, `dense_recovery_enabled` and `dense_recovery_allowed_actions` the same day
with their blocks.

## Files

| File | Holds |
| --- | --- |
| `service.py` | `AutonomousGraspService`: the factories, `pick()`, `set_prompt()`, the attempt, the decision gate, the recovery loop |
| `cells.py` | `build_real_cell`, `build_rehearsal_cell` and their component builders, `CellBuildRefused` |
| `config.py` | `GraspMode`, `resolve_grasp_mode`, `GraspBehaviorProfile`, `EffectiveGraspingConfig` and its per-phase parts |
| `report.py` | `AutonomousGraspOutcome`, `AutonomousGraspReport` |
| `prompt.py` | `PickPrompt` |
| `rehearsal.py` | the synthetic one-box scene a rehearsal perceives |
| `builders.py` | the per-phase wiring `from_robot_config` runs, `build_effective_config`, `apply_orchestrator_overlays` |
| `watchdog.py`, `latency.py` | drift and out-of-distribution events; rolling p95 latency, which never fails closed |
| `shadow.py`, `action_mask_eval.py` | the reinforcement-learning shadow router and its per-candidate action mask |
| `record_logging.py` | the report to `GraspAttemptRecord` serializer, and the process's background writer |

## Details

- Guide: [the pick loop](../../../../docs/guide/05-pick-loop.md); every `robot.grasping` block and the mode gate: [grasping-config-reference.md](../../../../docs/grasping-config-reference.md)
- The parent package: [execution](../README.md); the command line: [real_cell](../real_cell/README.md)
- What it drives: [grasping](../../grasping/README.md), [safety](../../safety/README.md), [grasping/rl](../../grasping/rl/README.md)
- Tests: `tests/test_autonomous_grasp_service.py`, `tests/test_autonomous_grasp_report_contract.py`, `tests/test_cell_builders.py`, `tests/test_pick_reports_a_cell_fault.py`, `tests/test_effective_config_field_split.py`, `tests/test_a_wrist_pick_hands_its_looks_to_the_pick_loop.py`, `tests/test_looks_come_from_the_cell_profile.py`, `tests/test_a_boxed_in_part_is_pushed_inside_the_pick.py`, `tests/test_a_jaw_count_nobody_vouches_for_moves_nothing_more.py`, `tests/test_a_pick_closes_along_the_axis_its_program_names.py`, `tests/test_every_grasp_closes_the_way_round_the_hand_naturally_stands.py`
