# The pick loop

`BinPickingOrchestrator` turns the tiers into one attempt. The calculator can already say why a pick
failed; this package turns those reasons into actions, and reports what the loop is doing while it
runs.

| Module | Owns |
|---|---|
| [`pick_loop.py`](pick_loop.py) | `BinPickingOrchestrator`, `PickReport`, `PickOutcome`: the loop itself |
| [`progress.py`](progress.py) | `PickStage`, `PickProgress`, `emit`: what a pick is doing while it does it |
| [`target_selector.py`](target_selector.py) | The clutter-aware selector: which object first, when several are viable |
| [`_pick_helpers.py`](_pick_helpers.py) | Two pure adapters: a chosen `GraspPoint` to the `GraspPose` the swept-volume validator wants, and per-segmentation successes to `TargetCandidate` records |
| [`_shadow_aggregator.py`](_shadow_aggregator.py) | The shadow-telemetry collaborator, described below |

The orchestrator depends only on the `RobotArm` Protocol, a calculator, a perception Protocol and an
optional viewpoint planner. Real perception and a real next-best-view solver plug in without touching
this file, which is how the simulation runners and the real cell differ from each other while sharing
the loop.

## Reasons become actions

| Reason from the calculator | What the loop does |
|---|---|
| `RESCAN_RECOMMENDED`, or no candidate at all | Request a fresh perception frame, without moving the robot |
| `ACTIVE_PERCEPTION_RECOMMENDED` | Ask a viewpoint planner for a new camera pose, drive there, re-acquire |
| `IK_FAILED` on the top candidate | Try the next ranked candidate before escalating |
| Anything else | Terminate with the corresponding outcome |

`PickOutcome` is the typed terminal: `EXECUTED`, `RESCANNED_EXHAUSTED`, `RELOCATED_EXHAUSTED`,
`NO_PERCEPTION`, `ABORTED`, `CANCELLED`, `OBJECT_NOT_DETECTED`, `EXECUTION_FAILED`,
`CAMERA_FRAME_REJECTED`, `NO_COMMIT_INSUFFICIENT_FUSION`, `APPROACH_PATH_BLOCKED`,
`CONTROLLER_NOT_OPERATIONAL`. The last of those is what stops the loop retrying into a controller
that has protective-stopped.

## Progress

`run()` is a single blocking call that returns one report at the end. A subscriber that attaches a
listener learns something while the pick is still running.

```
PICK_STARTED -> ATTEMPT_STARTED -> PERCEIVED -> RANKED -> EXECUTING -> ATTEMPT_FINISHED -> PICK_FINISHED
                                                 |                                          |
                                                 NO_CANDIDATE                               CANCELLED
```

Three properties, each of them deliberate.

Default off and byte identical. With no listener attached every emit is one `is None` test. The
payload is not constructed, no string formatted, nothing measured that was not measured already.
That is why `emit` takes the pieces rather than a built `PickProgress`: building the event only to
discard it is the cost this design exists to avoid.

Typed. The stage is a `StrEnum` and the payload a frozen dataclass, so the fields a console renders
cannot change shape unnoticed. The member set of `PickStage` is a fixed contract, because adding one
breaks a console replay buffer and any interface that maps stages to labels.

Emission cannot break a pick. Listener exceptions are swallowed. A browser that disconnects
mid-render, a subscriber with a bug or a full disk in a log listener must not abort a motion already
in flight.

## Which object first

`target_selector.py` runs after per-segmentation grasps are computed and before the executor takes
one, for the case where several objects are simultaneously viable. It estimates how much removing
each candidate would unblock the others and returns an `OrderingDecision` carrying the chosen index,
the supporting scores, the resolved mode (`single_best` or `clutter_aware`) and a typed reason
(`local_max`, `unlock_swap`, `guard_blocked_swap`, `no_candidates` or `disabled`).

It is pure: no input or output, no logging, no robot state. The defaults reduce to taking the best
grasp, byte for byte. `TargetOrderingConfig.enabled` is false, `unlock_weight` is 0.0, and all three
blocker-graph signals (`mask_adjacency_enabled`, `depth_only_enabled`, `corridor_overlap_enabled`)
are off.

The block runs; what it needs is a scene. Ordering only differs from taking the best grasp when two
or more objects are graspable at the same time and one genuinely blocks another. A scene of
well-separated graspable objects yields unlock scores of zero and degenerates to the local maximum,
and a scene of objects tight enough to block each other tends to offer nothing graspable. Enabling
the block on the wrong scene changes no decision and is not evidence that it does nothing.

## The shadow aggregator, and why it owns no state

`_shadow_aggregator.py` holds the reinforcement-learning observability logic that is interleaved with
the deterministic perceive, score, decide, commit, execute loop. It is stateless by design.

Every per-pick slot lives on the orchestrator as `field(init=False)`, because the
`AutonomousGraspService` shadow seam reads several of them through `getattr` and the runtime pick
path reads another directly. Moving a slot off the instance drops telemetry silently, with no error
anywhere. The aggregator therefore reads loop state through getter closures and returns telemetry,
and the orchestrator's thin same-named delegators assign their own slots. Return and assign means a
slot-write typo is a type error rather than a silently created new attribute.

Every shadow producer swallows its own exceptions by contract, so that a broken shadow can never
break a pick. That also means a permanently broken shadow looks exactly like one that is switched
off, which is why the aggregator logs.

## See also

- [`../README.md`](../README.md) for the tiers this loop wires together
- [`../../execution/autonomous_grasp/README.md`](../../execution/autonomous_grasp/README.md) for the
  service that builds and drives this orchestrator
- [`../motion/README.md`](../motion/README.md) for the execution policy the chosen grasp is handed to
- [`../recovery/README.md`](../recovery/README.md) and
  [`../closed_loop/README.md`](../closed_loop/README.md) for the two second chances
- [`../../../../api/README.md`](../../../../api/README.md) for the console that consumes `PickStage`
  over a WebSocket
- [`../../../../docs/grasping-config-reference.md`](../../../../docs/grasping-config-reference.md)
  for the `ordering` block and the mode gate
