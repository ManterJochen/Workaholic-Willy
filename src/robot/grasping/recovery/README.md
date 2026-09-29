# Recovery (`src.robot.grasping.recovery`)

What to do after a pick failed: look again, try another part, or, inside a declared envelope, nudge the
part or shake the bin. It plans; every motion it asks for still goes through the
same `RobotArm.move` and safety preflight as everything else. The loop a cell built from config runs
moves nothing, whatever it allows (see "What the service's loop does" below).

It ships off. You reach it through your cell's config, and the pick service wraps each attempt in the
recovery loop when it is on:

```yaml
robot:
  grasping:
    recovery:
      enabled: true
      allowed_actions: [rescan]   # the default is empty: enabled alone does nothing
      max_recovery_actions: 2
```

Call it directly only to test a failure class against your own policy:

```python
from src.robot.grasping import GraspFailureReason
from src.robot.grasping.recovery.orchestrator import RecoveryDispatcher
from src.robot.grasping.recovery.policy import SceneRecoveryAction

dispatcher = RecoveryDispatcher(overrides={GraspFailureReason.ALL_COLLIDED: (SceneRecoveryAction.RESCAN,)})
print(dispatcher.actions_for(GraspFailureReason.MOTION_PLAN_REFUSED))   # (RESCAN,) and nothing else
```

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `SceneRecoveryPolicy` | the pick service from `robot.grasping.recovery` | `permits(action)` | `bool` |
| `RecoveryDispatcher` | `RecoveryDispatcher(overrides=...)` | `actions_for(reason)` | the actions to try, in order |
| `RecoveryOrchestrator` | the pick service | `next_step(context)` | `SceneRecoveryPlan`, or `None` to stop |
| a plan | the orchestrator | `execute_recovery_motion(arm=..., plan=..., policy=...)` | `SceneRecoveryReport` |
| a failed pick | the pick service | `run_recovery_loop(...)` | the final report and the recovery trail |

## The actions

`NONE` is the typed "do nothing" and may not appear in an allow-list; `ABORT` hands over to the operator.
The four the dispatcher can plan, in rising order of how much they touch the world:

| Action | What it does | Needs |
| --- | --- | --- |
| `RESCAN` | perceive again from the same viewpoint | nothing |
| `NEXT_TARGET` | names a switch to another segmentation; nothing implements the switch since `NextTargetRecoveryStrategy` left on 2026-09-29, and no built-in profile allows it | a profile that lists it |
| `NUDGE_TARGET` | a small bounded push on the target | a `FixtureEnvelope` |
| `CONTAINER_AGITATE` | a bounded motion that redistributes a known container's contents | a `FixtureEnvelope` and a non-zero amplitude |

The first two change only what the cell knows; the last two change where things are, which is why
they need an envelope. The strategies are `SmallNudgeStrategy` and `ContainerAgitateStrategy`; the pick
service's loop plans from the dispatcher alone.

### What the service's loop does

The loop a cell built from config runs (`robot.grasping.recovery`, driven by the pick service) builds
its orchestrator with `bypass_strategies=True` and no strategies, so a plan names an action and carries
nothing else. Its outer gate is the mode's built-in profile, which `recovery.allowed_actions` cannot
widen: `auto` allows `rescan`, `dense_clutter` allows `rescan` and `nudge_target`, and `easy` allows
nothing. So only two actions can ever be planned there, and the other two never are:

- `rescan` runs the next pick, which perceives and ranks the scene afresh. Nothing excludes the part
  that failed.
- `nudge_target` (`dense_clutter` only, with a fixture) carries no offset, and
  `execute_recovery_motion` refuses it before the arm moves (`refused_no_offset`); the loop then ends
  `escalated_no_recovery`.
- `next_target` and `container_agitate` pass the load in `recovery.allowed_actions` and are never
  planned, because no built-in profile lists them. Naming them changes nothing.

So a config-built cell never pushes or shakes. A physical action moves the arm only for a caller that
widens the profile and builds the orchestrator with the strategy, as the simulator runner
`run_dense_pick --g6` does with `container_agitate` and `ContainerAgitateStrategy`.

`NEXT_VIEWPOINT` was merged into `RESCAN` on 2026-09-29: with the viewpoint planners gone nothing moved
the camera for it, so it re-perceived exactly as `RESCAN` does. A config or a preset that still names
`next_viewpoint` is refused at load, `removed on purpose: use rescan`; a record logged before then keeps
the string, and the offline recovery trainer still reads it. `ActivePerceptionRecoveryStrategy` (`RESCAN`,
then `NEXT_VIEWPOINT`), `NextTargetRecoveryStrategy` and `NoRecoveryStrategy` left the same day with the
`dense_recovery` block, the only thing that built them.

## Every gate an action passes

An action is planned only when all of these hold; otherwise nothing moves.

- The policy is enabled and the resolved mode is in its `apply_modes`.
- The attempt has not used `max_recovery_actions`, and the action is within its `per_action_budget`.
- The mode's behaviour profile lists it in `recovery_allowed_actions`, and the policy lists it too.
- It has not already been tried for the same failure class (`RecoveryHistoryEntry` records the pairs).
- A physical action has a `FixtureEnvelope`, and the policy refuses to be built without one.

`easy` never recovers, twice over: its profile allows no action, and it is absent from the default
`apply_modes` (`auto`, `dense_clutter`). `dense_clutter` is the one built-in profile that allows a push,
`nudge_target`, since `dense_autonomous` left on 2026-09-29, and the service's loop still refuses it
before it moves (above). `ContainerAgitateStrategy` adds its own gate:
it plans nothing until `max_agitate_amplitude_mm` is above zero, and no built-in profile lists
`container_agitate`. When armed, the agitation is three waypoints, each re-clamped to the envelope: an
air shake while `agitate_contact_depth_mm` is `0.0`, otherwise a descend, a sweep along +X and a retract.

## Failure class to action

`RecoveryDispatcher` maps each `GraspFailureReason` to actions; `overrides` replaces the sequence for any
class you list. Two entries are rules rather than tuning:

- `MOTION_PLAN_REFUSED` maps to `RESCAN` alone. Re-planning the same pose achieves nothing, and every
  other action either moves the arm or redirects the cell to a part nobody asked for. Moving after the
  cell declined to move is how a refusal becomes a motion.
- `CONTROLLER_NOT_OPERATIONAL` maps to nothing: a stopped cell needs a person, not a retry. So do
  `TARGET_LABEL_NOT_FOUND`, `TOPOLOGY_RISK_REJECTED`, `SEMANTIC_REJECTED` and
  `DEFORMABLE_ROUTING_REQUIRED`.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| refused at load, and `ValueError` from `SceneRecoveryPolicy` | `nudge_target` or `container_agitate` allowed with no fixture | declare `recovery.fixture` |
| refused at load | an unknown action or mode name | use the names in the tables above |
| `ValueError` from `SceneRecoveryPolicy` | `NONE` in `allowed_actions`, a duplicate, or a negative budget | fix the list |
| plan `NONE`, with a reason | a gate above failed | read the plan's `reason` and telemetry |
| report `refused_no_tcp`, `refused_no_offset`, `refused_agitate_disabled`, `refused_envelope_violation` | no start pose, a nudge with no offset (every nudge the service's loop plans), an agitation with no amplitude (one planned without `ContainerAgitateStrategy`; the service's loop plans none), or a waypoint outside the envelope | nothing moved; fix the envelope or the call |
| report `aborted_motion_failed` | a recovery move came back other than `EXECUTED` | nothing further moves; read the motion status |

## Traps

- `robot.grasping.recovery` is the loop around the whole attempt, and the only recovery block since
  `robot.grasping.dense_recovery` was removed on 2026-09-29 (a tree that still writes it is refused).
- `allowed_actions` is empty in the config schema and `(RESCAN,)` on a runtime `SceneRecoveryPolicy`
  built by hand. Enabling recovery in YAML without naming actions allows nothing.
- A mode outside `apply_modes` keeps recovery inert whatever `recovery.enabled` says, and a list that
  still names `closed_loop` or `dense_autonomous` (removed 2026-09-29) is refused at load.
- `next_target` and `container_agitate` in `recovery.allowed_actions` load without a word and do
  nothing: the mode's profile, which no config key widens, lists neither.

## Status

| Capability | Evidence |
| --- | --- |
| The recovery loop and container agitation | measured in simulation: `run_dense_pick` drives the service's loop with `--recovery`, and agitation with `--g6` through a strategy it builds itself |
| A nudge from a cell built from config | refused before it moves, by construction (above) |
| An agitation or a `next_target` from a cell built from config | never planned: no built-in profile lists either (above) |
| A nudge or an agitation on a physical cell | never touched hardware |

## Files

| File | Holds |
| --- | --- |
| `policy.py` | `SceneRecoveryAction`, `SceneRecoveryPolicy`, `FixtureEnvelope`, the strategies, `execute_recovery_motion` |
| `orchestrator.py` | `RecoveryDispatcher`, `RecoveryOrchestrator`, `run_recovery_loop`, `RecoveryHistoryEntry` and the trail |
| `trail_serialize.py` | `recovery_actions_from_trail`, the trail as the record's `recovery_actions` block |

## Details

- [`loop/`](../loop/README.md) reports the failure this package reacts to, and
  [`types/`](../types/README.md) holds `GraspFailureReason`.
- `closed_loop/` held the other second chance, the two-scan refinement taken before the failure; it
  was removed on 2026-09-29.
- [`safety/`](../../safety/README.md) holds the guards every recovery waypoint still passes.
- [The grasping config reference](../../../../docs/grasping-config-reference.md) for the `recovery`
  block and the modes it fires in.
- Tests: `tests/test_recovery_policy.py`, `tests/test_t5_recovery_dispatcher.py`,
  `tests/test_t5_recovery_orchestrator.py`, `tests/test_t5_service_retry_loop.py`,
  `tests/test_recovery_fixture_wiring.py`.
