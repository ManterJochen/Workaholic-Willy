# Recovery (`src.robot.grasping.recovery`)

What to do after a pick failed: look again, look from elsewhere, try another part, or, inside a declared
envelope, nudge the part or shake the bin. It plans; every motion it asks for still goes through the
same `RobotArm.move` and safety preflight as everything else.

It ships off. You reach it through your cell's config, and the pick service wraps each attempt in the
recovery loop when it is on:

```yaml
robot:
  grasping:
    recovery:
      enabled: true
      allowed_actions: [rescan, next_viewpoint]   # the default is empty: enabled alone does nothing
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
The five a strategy can plan, in rising order of how much they touch the world:

| Action | What it does | Needs |
| --- | --- | --- |
| `RESCAN` | perceive again from the same viewpoint | nothing |
| `NEXT_VIEWPOINT` | move the camera and perceive again | a viewpoint planner |
| `NEXT_TARGET` | switch to another segmentation in the frame already captured | at least two segmentations |
| `NUDGE_TARGET` | a small bounded push on the target | a `FixtureEnvelope` |
| `CONTAINER_AGITATE` | a bounded motion that redistributes a known container's contents | a `FixtureEnvelope` and a non-zero amplitude |

The first three change only what the cell knows; the last two change where things are, which is why
they need an envelope. The strategies are `NoRecoveryStrategy`, `ActivePerceptionRecoveryStrategy`
(`RESCAN`, then `NEXT_VIEWPOINT`), `NextTargetRecoveryStrategy`, `SmallNudgeStrategy` and
`ContainerAgitateStrategy`.

## Every gate an action passes

An action is planned only when all of these hold; otherwise nothing moves.

- The policy is enabled and the resolved mode is in its `apply_modes`.
- The attempt has not used `max_recovery_actions`, and the action is within its `per_action_budget`.
- The mode's behaviour profile lists it in `recovery_allowed_actions`, and the policy lists it too.
- It has not already been tried for the same failure class (`RecoveryHistoryEntry` records the pairs).
- A physical action has a `FixtureEnvelope`, and the policy refuses to be built without one.

`easy` never recovers, twice over: its profile allows no action, and it is absent from the default
`apply_modes` (`auto`, `dense_clutter`, `dense_autonomous`). `ContainerAgitateStrategy` adds its own gate:
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
| report `refused_no_tcp`, `refused_agitate_disabled`, `refused_envelope_violation` | no start pose, no amplitude, or a waypoint outside the envelope | nothing moved; fix the envelope or the call |
| report `aborted_motion_failed` | a recovery move came back other than `EXECUTED` | nothing further moves; read the motion status |

## Traps

- `robot.grasping.recovery` is the loop around the whole attempt. `robot.grasping.dense_recovery` is a
  separate block with its own `enabled`, and switching one on does not switch on the other.
- `allowed_actions` is empty in the config schema and `(RESCAN, NEXT_VIEWPOINT)` on a runtime
  `SceneRecoveryPolicy` built by hand. Enabling recovery in YAML without naming actions allows nothing.
- The `verification_heavy` preset enables recovery in `closed_loop` mode but keeps the default
  `apply_modes`, which leave `closed_loop` out. Recovery stays inert there until a cell adds the mode.

## Status

| Capability | Evidence |
| --- | --- |
| The recovery loop and container agitation | measured in simulation: `run_dense_pick` drives the service's loop with `--recovery` and agitation with `--g6` |
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
- [`closed_loop/`](../closed_loop/README.md) is the other second chance, taken before the failure.
- [`safety/`](../../safety/README.md) holds the guards every recovery waypoint still passes.
- [The grasping config reference](../../../../docs/grasping-config-reference.md) for the `recovery` and
  `dense_recovery` blocks and the modes they fire in.
- Tests: `tests/test_recovery_policy.py`, `tests/test_t5_recovery_dispatcher.py`,
  `tests/test_t5_recovery_orchestrator.py`, `tests/test_t5_service_retry_loop.py`,
  `tests/test_recovery_fixture_wiring.py`.
