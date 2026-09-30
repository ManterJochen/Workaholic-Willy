# The pick loop (`src.robot.grasping.loop`)

`BinPickingOrchestrator` turns the grasping tiers into one attempt: perceive, rank, choose a target,
execute, and act on why an attempt failed. On a wrist camera it first **looks around**: it visits the
pick's looks, fuses what each one saw and stops as soon as the grasp is safe. It also reports what a
pick is doing while it runs.

The pick service builds and drives it; you do not construct one. What you can attach is a listener that
hears each stage. This runs at a desk, on the dummy arm of the `console_dummy` profile:

```python
from willy import Cell, PickRun, Recording, load_tree
from src.robot.grasping.loop.progress import PickProgress


def show(event: PickProgress) -> None:
    print(event.stage, event.attempt, event.candidate_count)


cell = Cell.rehearsal(load_tree("console_dummy").robot)   # a dummy arm and a synthetic scene
cell.build().attach_progress_listener(show)               # the service that drives the loop
print(PickRun.from_cell(cell, runs=1, recording=Recording.off()).execute())
```

It prints `pick_started`, `attempt_started`, `perceived`, `ranked` with the candidate count,
`executing`, `attempt_finished` and `pick_finished`, then the run's verdict.

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `BinPickingOrchestrator` | the pick service | `run()` | `PickReport` with a `PickOutcome` |
| a listener | your function taking a `PickProgress` | `service.attach_progress_listener(fn)` | events while the pick runs |
| `TargetOrderingConfig` | `robot.grasping.ordering` | `select_target(candidates=..., config=...)` | `OrderingDecision` |
| `LookedAround` | `orch.look_around()`, on a wrist camera | `orch.go_on_with(looked)`; read `orch.looked_around` after the pick | the looks visited, the grasp they judged, the looks refused, the generated view, the move back |
| the recovery hand-over | the pick service, per pick | `orch.push_gate`, `orch.exclusion_zones`; read `orch.pushes` and `orch.failed_part` after the pick | what each push the pick considered came to (`PickPush`), and the part it failed on, which `next_target` skips |

The orchestrator depends only on the `RobotArm` Protocol, a calculator and a perception Protocol, which
is how the simulation runners and a real cell share one loop.

## Reasons become actions

| Reasons from the calculator | What the loop does |
| --- | --- |
| `RESCAN_RECOMMENDED`, `NO_CANDIDATES_GENERATED`, `NO_VALID_DEPTH`, `LOW_DEPTH_CONFIDENCE`, `LOW_MASK_CONFIDENCE`, `ACTIVE_PERCEPTION_RECOMMENDED`, `ALL_OUT_OF_WORKSPACE` | a fresh frame, without moving the robot; a wrist pick handed looks ends the attempt `exhausted` instead, because its next view was its next look |
| anything else | ends with the matching outcome |

A fixed camera never moves to look again. Another view comes from a second camera
([multiview/](../multiview/README.md)) or, on a wrist camera, from the looks a program hands the pick
(`src/robot/execution/looks.py`), below. The viewpoint planners and the relocate path were removed on
2026-09-29.

With an IK service wired, the calculator drops an unreachable candidate and the next ranked one stands;
without one, the motion planner is the first to refuse it. `IK_FAILED` means every candidate was
unreachable, and it arrives with `RESCAN_RECOMMENDED`. With the swept-path validator on, the loop executes the first candidate whose
approach is clear.

`PickOutcome` is the typed end: `EXECUTED`, `RESCANNED_EXHAUSTED`, `NO_PERCEPTION`,
`ABORTED`, `CANCELLED`, `OBJECT_NOT_DETECTED`, `EXECUTION_FAILED`, `CAMERA_FRAME_REJECTED`,
`APPROACH_PATH_BLOCKED`, `CONTROLLER_NOT_OPERATIONAL`, `GRIPPER_FAULT`. `RELOCATED_EXHAUSTED` left with the
relocate path; a record it ended reads `no_valid_grasp`.
`CANCELLED` is an operator's stop between attempts, kept apart from `ABORTED` so a stop button never counts
as a failed execution. `CONTROLLER_NOT_OPERATIONAL` stops the loop retrying into a controller that has
protective-stopped. `GRIPPER_FAULT` is a hand that needs a person: a gripper that raised while it was
commanded, a toggle with no sensor that would not start on jaws it believes closed, or a hand a push found
nobody can vouch for (a toggle's count, a width-measuring gripper not connected or unreadable); a campaign
stops on it.

`PickAttempt.action` says what each attempt did:

| Action | When |
| --- | --- |
| `executed`, `object_not_detected`, `camera_frame_rejected`, `approach_path_blocked`, `gripper_fault`, `execution_failed` | the execution policy's outcome |
| `rescan`, `exhausted` | no candidate stood: a fresh frame follows, or the pick ends |
| `controller_not_operational` | the attempt did not execute and the controller cannot move; on the way to a look it reads `look_refused` |
| `look_refused` | a wrist pick reached none of its looks; a motion to a look, to the generated view or back failed once it may have been commanded, or was refused by its verb before any command; or the controller stopped on the way (outcome `CONTROLLER_NOT_OPERATIONAL`). Nothing was perceived there and nothing else was commanded |
| `faces_unseen` | `both_faces` was asked and no view showed both jaw contact faces of the chosen grasp; `PickAttempt.withheld` names the face |
| `push` | a push inside a wrist camera's attempt (`dense_clutter`, below) pushed the part, stopped where the arm stands, or found before anything moved that the controller cannot move (the pick ends `CONTROLLER_NOT_OPERATIONAL`) or that nobody can vouch for the hand, a toggle's count or a width-measuring gripper not connected or unreadable (the pick ends `GRIPPER_FAULT`); `PickAttempt.push` carries its code |

`relocate` appears only on attempts recorded before 2026-09-29.

## Which way the jaws close

Every object's calculator result passes `_closing_along` right after `compute_result`, before the target
choice, the looks' judgement and jaw faces, the deep-ranker stamp, the reranks, the approach check and
the push read its candidates. So the grasp judged is the grasp executed.

- **A program's closing axis** (`GraspMotion(closing_axis=...)`, on the policy as `closing_axis`) keeps
  only the grasps heading within 30 deg of it, either way round, each turned half a turn about its
  approach where that puts it the named way round (the owner's "choose, don't twist"). While it is set
  the calculator is asked for 36 candidates (`CANDIDATES_THE_AXIS_CHOOSES_AMONG`, or its own cap where
  that is more), the best of its own cap along the axis are kept, and the cap is given back, raise or
  not. None left is `NO_VALID_GRASP` with a WARNING that names the axis and the 30 deg, and
  `PickAttempt.withheld` says it on every attempt that found no grasp, also when the frame failed on
  another part. A part the axis refused is the frame's failure only where no other part failed for its
  own reason, whichever order the camera lists them in.
- **The cell's natural orientation** (`natural_closing_axis`, set from `robot.natural_closing_axis` by
  `RuntimePickService.from_robot_config`) turns every grasp, where no program names an axis, to the wrist
  half-turn whose tool +X lies nearer it; none is left out or tilted. Beside the simulator's twist
  (`align_closing_to_base_x`, any true value) it turns no grasp, while a push still takes its way round,
  and a value set by hand that names no axis is refused by `run()` and `look_around()` before anything
  moves, naming the field.
- **The overlay.** Where either changed a result, the calculator that computed it draws its overlay again
  over the grasps kept (`redraw_debug_image`), or drops it where it cannot, so the picture a campaign
  pins ranks first the grasp gripped.
- **Refused before anything moves** (`ValueError`, `TypeError`): a `closing_axis` beside the twist, one
  set on the policy later that names no axis, and `both_faces` beside the twist.

## A wrist camera looks around

A camera on the wrist (its frames placed by the tool pose, an `EyeInHandFrameResolver`) sees only what
the arm points it at. The service hands each pick its **looks** (`orch.looks`: `JointPositions` or
`"home"`, in order), and the loop looks from each one before its first attempt. The goal is a **safe
grasp point**, never a whole scan of the part.

```mermaid
flowchart LR
    M["move to the next look"] --> P["perceive, fuse with<br/>every earlier look"]
    P --> J{"valid grasp,<br/>no rescan reason?"}
    J -->|yes| A["approach"]
    J -->|"no, looks left"| M
    J -->|"no, looks used up"| G["one generated view,<br/>straight joint line only"]
    G -->|"a valid grasp"| B["back to the look that ranked it,<br/>straight joint line only"]
    G -->|"none"| X["exhausted"]
    B --> A
```

- **Early stop.** Each look is fused with every earlier look and, where `grasping.fusion.geometry` is on,
  with the fixed cameras, acquired once per pick. The grasp is computed again on what they saw together,
  and the looking stops at the first look whose grasp is valid with no rescan reason. An uncertain grasp
  or no candidate goes on to the next look.
- **One part.** Once a look ranks a valid grasp, its part is kept: later looks find it by association,
  and a look that does not see it is left out and said. Looks that call the part by different labels
  make its grasp uncertain (`RESCAN_RECOMMENDED`); looks that agree change nothing. One INFO account per
  pick handed looks says where they ended (a safe stop, no grasp, a withheld grasp or the fall-back) and
  ends `label agreed in N of M looks`: M the looks whose views of the part are in its cloud, N those that
  call it what the judged look calls it (`association.label_agreement_said`). Nothing acts on it.
- **All the data.** The support plane is refined from the part's cloud fused over two or more looks, and
  the neighbours every look saw reach the candidate filter, so a part only an earlier look saw stays an
  obstacle.
- **A refused look** (a guard or the planner refused it before anything was sent) is skipped with a
  WARNING and counts as used up. A pick that reaches none of its looks ends `look_refused` with nothing
  perceived. A look motion that failed once it may have been commanded, or a controller that stopped,
  ends the looking there with nothing else commanded.
- **One generated view**, the last resort. Once every declared look is used up and the grasp is still
  not safe, the loop turns the judged look about the part toward the jaw contact face of the chosen grasp
  no view showed, or toward the side no look faced when there is no grasp. Every turn is screened by the
  arm before it moves (`nearest_configuration`), smallest first: at most 150 deg of travel on any joint,
  every joint within half a turn of home (the cable window), at most 3 motions. It drives the **straight
  joint line only**: never cuRobo, no retract, no detour, and a blocked line skips that turn. A move past
  120 deg logs a WARNING once the arm stands there. It runs only against the planner world that holds
  every frame of the pick; otherwise, and on an arm without the verbs (the Isaac arm, the dummy), no view
  is generated and an INFO line says why (`no view is generated: ...`).
- **The move back.** Where the grasp was ranked on another look than the one the arm stands at, the arm
  goes back there on the straight joint line, under the same 150 deg cap and 120 deg warning. Refused, a
  WARNING says so and the approach starts from where the arm stands, planned and judged as every approach
  is.
- **Looks used up.** The pick goes on with the valid grasp the looks judged, safe or not, and says so;
  no valid grasp ends the attempt `exhausted`.
- **Held frames.** Every frame the wrist camera takes is held in the arm's live planner world from the
  first look until the pick ends, whether it gripped, failed or raised (`close_looks`, called by
  `run()`; `look_around()` lets go itself when it raises). Where two or more views make up the part's
  cloud, that fused cloud is what the world leaves out for the pick and where the part is said to be
  (`PickReport.target_centre_mm`). A pick handed no looks offers its own frame for both, even where a
  fixed camera is fused into its grasp.
- **`both_faces`**, off by default: a look is good only once both jaw contact faces of the chosen grasp
  were seen. Where no view, declared or generated, showed both, nothing is gripped: `faces_unseen`,
  `RESCANNED_EXHAUSTED` (the service's `no_valid_grasp`), naming the face. A fixed camera grips only a
  grasp its cameras (fused, where the cell fuses them) showed both faces of. The grasp gripped is the
  judged one: the reranks stand down and the approach check tries no other candidate. A suction cup has
  no jaw faces and always passes. The way round comes from the natural orientation and one axis to close
  along from a `closing_axis` (above): both act before the faces are judged, and the refusal then names the
  jaw after the turn. `both_faces` with `GraspMotion(align_closing_to_base_x=True)` raises `ValueError`
  before anything moves: that yaw closes the jaws on faces nobody judged.
- **The hand-eye check** runs once, on the judgement the looking ends on: the median distance between
  the looks' surfaces of the part. Above `HAND_EYE_DRIFT_WARN_MM` (6 mm, in
  [`multiview/association.py`](../multiview/association.py)) a WARNING says the calibration may have
  drifted. Nothing else is done about it.
- **Handed on.** `look_around()` is a wrist camera's and raises `ValueError` on a fixed camera. The
  decision path decides on its judgement and hands it to the next `run()` only through
  `go_on_with(looked)`. A wrist pick handed no looks looks from where the arm stands and rescans there, as
  it always did.

## Recovery inside the attempt

Two recovery actions reach into the loop, and only where the service arms them
([recovery/](../recovery/README.md)):

- **`next_target`** skips a failed part: beside the label gate, the loop passes over a segmentation of
  the same label whose centre lies in an exclusion zone. With only excluded parts left, the pick stops
  and says so. On a wrist camera the new part gets the look sequence again: the early stop, at most one
  generated view. Before it drives the looks again, the service reads a toggle's count, and one nobody
  can vouch for ends the pick as a gripper fault before any look is driven.
- **The push** (`nudge_target`, `dense_clutter` only) runs on a wrist camera's pick, after its looks
  judged the part; a fixed camera never pushes. It is due when no candidate survived and at least one
  collided (`ALL_COLLIDED`), or, where approach validation runs, when every approach was blocked; in both
  cases a neighbour was seen within 25 mm of the part. The loop plans the push from the table and the
  neighbours every look saw, drives it, goes back to the look like a grasp approach (to the view it
  generated on the straight joint line alone), looks again and judges again. Never on `NO_VALID_GRASP` or
  `NO_CANDIDATES_GENERATED`: nothing there says a neighbour is in the way, and a part the closing axis
  refused carries only `NO_VALID_GRASP`, so it is never pushed, while a boxed-in neighbour of it still is.
  The push reads the frame's failing part as the closing axis and the natural turn left it, and closes the
  way round nearer the natural orientation where the cell names one (`execute_push(...,
  natural_closing_axis=)`). A push refused before anything
  moved lets the attempt go on, except two that end the pick with nothing commanded: a controller that
  cannot move (`CONTROLLER_NOT_OPERATIONAL`) and a hand nobody can vouch for, a toggle's count or a
  width-measuring gripper not connected or unreadable (`GRIPPER_FAULT`, the owner's rule of 2026-09-30). A
  push that stopped once something may have moved, a toggle's count nobody can vouch for before a contact
  leg or once the arm is up among it, ends the pick where the arm stands (`ABORTED`, which the service
  reports `unsafe_recovery_refused`, or `CONTROLLER_NOT_OPERATIONAL`). The events carry it: `ATTEMPT_FINISHED` with action `push` and `push`,
  `push_mm`, `push_reason`, `push_leg` or `looked_again` in `extra`, and a `NO_CANDIDATE` after a push
  refused before motion with `push` and `push_reason`, or the zones' sentence as `excluded`.

## Progress

`run()` is one blocking call; a listener learns something while it is still running. `PickStage` is a
fixed set: `PICK_STARTED`, `ATTEMPT_STARTED`, `PERCEIVED`, `RANKED`, `NO_CANDIDATE`, `EXECUTING`,
`ATTEMPT_FINISHED`, `PICK_FINISHED`, `CANCELLED`. The operator console turns each into a sentence, so a
new member is a change to it as well.

- Off and byte-identical by default: with no listener each emit is one `is None` test, and no event is
  built.
- Typed: the stage is a `StrEnum` and the event a flat frozen dataclass, so it serialises with no
  translation layer.
- It cannot break a pick: a listener that raises is logged and ignored. It is called inline on the
  thread driving the robot, so a slow listener slows the loop; hand the event to a queue and return.

`service.set_cancel_check(fn)` lets a caller stop a run between attempts. It never interrupts a motion
in flight; the physical stop button does that.

## Which object first

`target_selector.py` runs when several objects are graspable at once. It estimates how much removing
each one would unblock the others and returns an `OrderingDecision`: the chosen index, the scores, the
mode (`single_best` or `clutter_aware`) and a reason (`local_max`, `unlock_swap`, `guard_blocked_swap`,
`no_candidates` or `disabled`). It is pure: no input or output and no robot state.

The defaults reduce to taking the best grasp, byte for byte: `enabled` is false and `unlock_weight` is
`0.0`, so the blocker graph is built and multiplied by zero. Of the three blocker signals,
`mask_adjacency_enabled` and `depth_only_enabled` work when switched on; `corridor_overlap_enabled` is
accepted and does nothing. Ordering can only change a decision where two graspable objects block each
other, and with a target label the choice is made before ordering is asked.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `CAMERA_FRAME_REJECTED` | the chosen grasp is still in the camera frame on a cell that requires BASE | check the camera's declared calibration |
| `APPROACH_PATH_BLOCKED` | every ranked candidate's approach sweep hits the scene cloud | clear the scene or re-perceive |
| `CONTROLLER_NOT_OPERATIONAL` | the controller is stopped or powered off | a person clears the stop |
| `GRIPPER_FAULT` | the gripper raised, a toggle hand believes its jaws closed and nobody at a terminal said otherwise, or a push found nobody can vouch for the hand (a toggle's count, a width-measuring gripper not connected or unreadable) | look at the hand, then run from a terminal or connect again |
| `RESCANNED_EXHAUSTED`, `NO_VALID_GRASP`, `withheld` naming the axis | the program's `closing_axis` left the part no grasp within 30 deg | name an axis the part's grasps close along, or none |
| `RuntimeError` | a configured fused camera delivered no frame and `on_camera_unavailable` is `refuse` | see [`multiview/`](../multiview/README.md) |
| `look_refused`, `EXECUTION_FAILED` or `CONTROLLER_NOT_OPERATIONAL` | a wrist pick reached none of its looks, a look motion may have moved the arm, or the controller stopped on the way | read the motion status on the attempt; teach looks the planner admits; a stopped controller needs a person |
| `faces_unseen`, `RESCANNED_EXHAUSTED` | `both_faces` was asked and no view showed both jaw contact faces | give the pick a look that faces the missing side, or leave the switch off |
| `ValueError`, nothing moved | `look_around()` on a fixed camera; `both_faces` or a `closing_axis` with a motion that yaws every grasp (`align_closing_to_base_x`); a `closing_axis` or a hand-set `natural_closing_axis` that names no axis (`TypeError` for a value of another type); `go_on_with` of a look sequence that is not the last one open | fix the call |

## Status

| Capability | Evidence |
| --- | --- |
| The loop, its outcomes and the progress events | measured in simulation: every Isaac pick runs through it |
| `CONTROLLER_NOT_OPERATIONAL` | measured against real controller software: a protective stop in URSim |
| Clutter-aware ordering | never touched hardware: tested on synthetic scenes, no measured scene where order matters |
| The wrist looks, the generated view, the move back and `both_faces` | pinned by tests on fake arms and cameras, and the Isaac arm generates no view; never touched hardware |
| The push inside a wrist camera's attempt | pinned by tests with a fake arm and a fake live world; never touched hardware |

## Files

| File | Holds |
| --- | --- |
| [`pick_loop.py`](pick_loop.py) | `BinPickingOrchestrator`, `PickReport`, `PickOutcome`, `PickAttempt`, `LookedAround`, `judged_faces_turned_away` |
| [`progress.py`](progress.py) | `PickStage`, `PickProgress`, `emit` |
| [`target_selector.py`](target_selector.py) | `select_target`, `TargetOrderingConfig`, `OrderingDecision`, the blocker graph |
| [`_pick_helpers.py`](_pick_helpers.py) | a chosen `GraspPoint` to a `GraspPose`, and successes to `TargetCandidate` records |
| [`_shadow_aggregator.py`](_shadow_aggregator.py) | the observe-only telemetry the loop emits for the learning tools |

The shadow aggregator owns no state: every per-pick slot stays on the orchestrator, because the pick
service reads several of them by name, and moving one would drop telemetry with no error. Every shadow
producer swallows its own exceptions so it cannot break a pick, which is why the aggregator logs.

## Details

- [`execution/autonomous_grasp/`](../../execution/autonomous_grasp/README.md) builds and drives this
  orchestrator; [`motion/`](../motion/README.md) executes the chosen grasp.
- [`execution/`](../../execution/README.md) holds the looks (`looks.py`) and the one generated view with
  its move back (`generated_view.py`); [`multiview/`](../multiview/README.md) fuses the looks and says
  which jaw contact faces were seen.
- [`recovery/`](../recovery/README.md) is the second chance. The post-grasp verification stage, which no
  pick path ran, left on 2026-09-29 with `closed_loop/`; the hold is the execution policy's own check
  after its close ([`motion/`](../motion/README.md)).
- [The console](../../../../api/README.md) streams `PickStage` over a WebSocket.
- [Guide 05, the pick loop](../../../../docs/guide/05-pick-loop.md), and
  [the grasping config reference](../../../../docs/grasping-config-reference.md) for `ordering`.
- Tests: `tests/test_pick_loop.py`, `tests/test_pick_progress.py`,
  `tests/test_pick_loop_controller_state.py`, `tests/test_t4_target_selector.py`,
  `tests/test_t4_pick_loop_ordering.py`, `tests/test_a_wrist_pick_looks_until_its_grasp_is_safe.py`,
  `tests/test_a_wrist_pick_runs_end_to_end.py`, `tests/test_both_jaw_faces_are_seen.py`,
  `tests/test_the_generated_view_moves_on_its_straight_joint_line_only.py`,
  `tests/test_a_boxed_in_part_is_pushed_inside_the_pick.py`,
  `tests/test_a_pick_closes_along_the_axis_its_program_names.py`,
  `tests/test_every_grasp_closes_the_way_round_the_hand_naturally_stands.py`,
  `tests/test_the_looks_say_how_many_agreed_on_the_label.py`,
  `tests/test_a_part_the_axis_refused_is_never_pushed.py`,
  `tests/test_a_jaw_count_nobody_vouches_for_moves_nothing_more.py`.
