# The pick loop (`src.robot.grasping.loop`)

`BinPickingOrchestrator` turns the grasping tiers into one attempt: perceive, rank, choose a target,
execute, and act on why an attempt failed. It also reports what a pick is doing while it runs.

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

The orchestrator depends only on the `RobotArm` Protocol, a calculator, a perception Protocol and an
optional viewpoint planner, which is how the simulation runners and a real cell share one loop.

## Reasons become actions

| Reasons from the calculator | What the loop does |
| --- | --- |
| `RESCAN_RECOMMENDED`, `NO_CANDIDATES_GENERATED`, `NO_VALID_DEPTH`, `LOW_DEPTH_CONFIDENCE`, `LOW_MASK_CONFIDENCE` | a fresh frame, without moving the robot |
| `ACTIVE_PERCEPTION_RECOMMENDED`, `ALL_OUT_OF_WORKSPACE` | the viewpoint planner's next camera pose, then a fresh frame; with no planner, a rescan |
| anything else | ends with the matching outcome |

With an IK service wired, the calculator drops an unreachable candidate and the next ranked one stands;
without one, the motion planner is the first to refuse it. `IK_FAILED` means every candidate was
unreachable, and it arrives with `RESCAN_RECOMMENDED`. With the swept-path validator on, the loop executes the first candidate whose
approach is clear.

`PickOutcome` is the typed end: `EXECUTED`, `RESCANNED_EXHAUSTED`, `RELOCATED_EXHAUSTED`, `NO_PERCEPTION`,
`ABORTED`, `CANCELLED`, `OBJECT_NOT_DETECTED`, `EXECUTION_FAILED`, `CAMERA_FRAME_REJECTED`,
`NO_COMMIT_INSUFFICIENT_FUSION`, `APPROACH_PATH_BLOCKED`, `CONTROLLER_NOT_OPERATIONAL`, `GRIPPER_FAULT`.
`CANCELLED` is an operator's stop between attempts, kept apart from `ABORTED` so a stop button never counts
as a failed execution. `CONTROLLER_NOT_OPERATIONAL` stops the loop retrying into a controller that has
protective-stopped. `GRIPPER_FAULT` is a hand that needs a person: a gripper that raised while it was
commanded, or a toggle with no sensor that would not start on jaws it believes closed; a campaign stops on
it.

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
| `NO_COMMIT_INSUFFICIENT_FUSION` | the commit gate found too little multi-view evidence and the re-look budget is spent | add a view, or leave the gate off |
| `APPROACH_PATH_BLOCKED` | every ranked candidate's approach sweep hits the scene cloud | clear the scene or re-perceive |
| `CONTROLLER_NOT_OPERATIONAL` | the controller is stopped or powered off | a person clears the stop |
| `GRIPPER_FAULT` | the gripper raised, or a toggle hand believes its jaws closed and nobody at a terminal said otherwise | look at the hand, run from a terminal |
| `RuntimeError` | a configured fused camera delivered no frame and `on_camera_unavailable` is `refuse` | see [`multiview/`](../multiview/README.md) |

## Status

| Capability | Evidence |
| --- | --- |
| The loop, its outcomes and the progress events | measured in simulation: every Isaac pick runs through it |
| `CONTROLLER_NOT_OPERATIONAL` | measured against real controller software: a protective stop in URSim |
| Clutter-aware ordering | never touched hardware: tested on synthetic scenes, no measured scene where order matters |

## Files

| File | Holds |
| --- | --- |
| [`pick_loop.py`](pick_loop.py) | `BinPickingOrchestrator`, `PickReport`, `PickOutcome`, `CommitPolicy` |
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
- [`recovery/`](../recovery/README.md) and [`closed_loop/`](../closed_loop/README.md) are the two second
  chances.
- [The console](../../../../api/README.md) streams `PickStage` over a WebSocket.
- [Guide 05, the pick loop](../../../../docs/guide/05-pick-loop.md), and
  [the grasping config reference](../../../../docs/grasping-config-reference.md) for `ordering`.
- Tests: `tests/test_pick_loop.py`, `tests/test_pick_progress.py`,
  `tests/test_pick_loop_controller_state.py`, `tests/test_t4_target_selector.py`,
  `tests/test_t4_pick_loop_ordering.py`.
