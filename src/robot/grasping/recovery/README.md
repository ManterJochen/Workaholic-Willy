# Recovery (`src.robot.grasping.recovery`)

What to do after a pick failed. This package is a planner, not a driver: it returns a typed plan
describing what should happen next, and the motion side goes through the same
`SafetyPreflight`-gated `RobotArm.move` surface as everything else.

Off by default. The loop runs only when `robot.grasping.recovery.enabled` is true; otherwise the
service short-circuits to a single, byte-identical pick.

## What this package guarantees

The `easy` mode never produces recovery motion, and it is guarded twice. The `easy` behaviour
profile carries an empty `recovery_allowed_actions` tuple, so every strategy short-circuits to
`NONE` whatever the policy says, and `easy` is also absent from the default `apply_modes`
(`auto`, `dense_clutter`, `dense_autonomous`). One guard would be enough to be correct; the second
is what holds through a config edit nobody reviewed.

A physical action without a `FixtureEnvelope` refuses to be constructed. Not refused at run time,
refused at build time: `SceneRecoveryPolicy.__post_init__` raises when `nudge_target` or
`container_agitate` appears in `allowed_actions` with no envelope, and the config schema raises the
same way one layer up. That is what stops a future config drift from quietly enabling motion outside
a known-safe workspace.

Bounded and anti-loop. `max_recovery_actions` caps the attempt, `per_action_budget` caps each
action, and `RecoveryHistoryEntry` carries `(action, failure class, outcome)` so the same action
cannot be tried forever against the same failure.

## The public surface

| Module | Owns |
| --- | --- |
| `policy.py` | `SceneRecoveryAction`, `SceneRecoveryPolicy`, `FixtureEnvelope`, `SceneRecoveryContext`, `SceneRecoveryPlan`, `SceneRecoveryReport`, the five strategies, and `execute_recovery_motion` |
| `orchestrator.py` | `RecoveryDispatcher` (the pure failure-class to action mapping), `RecoveryOrchestrator`, `run_recovery_loop`, and the history and trail bookkeeping |
| `trail_serialize.py` | `recovery_actions_from_trail`, the typed trail serialised for the record's `recovery_actions` block |

### The action vocabulary

`SceneRecoveryAction` has seven members. `NONE` is the typed do-nothing sentinel and is rejected if
it appears in an allow-list; `ABORT` escalates to the operator. The five that a strategy can plan,
in ascending order of how much they touch the world:

| Action | What it does | Needs |
| --- | --- | --- |
| `RESCAN` | perceive again from the same viewpoint | nothing |
| `NEXT_VIEWPOINT` | move the camera and perceive again | a viewpoint planner |
| `NEXT_TARGET` | switch to another segmentation in the frame already captured | a frame with at least two segmentations |
| `NUDGE_TARGET` | a small bounded push on the target | a `FixtureEnvelope` |
| `CONTAINER_AGITATE` | a bounded motion that redistributes a known container's contents | a `FixtureEnvelope` and a non-zero amplitude |

The first three change only what the cell knows. The last two change where things are, which is why
they are gated differently.

Five strategies ship: `NoRecoveryStrategy`, `ActivePerceptionRecoveryStrategy` (escalates `RESCAN`
then `NEXT_VIEWPOINT`), `NextTargetRecoveryStrategy`, `SmallNudgeStrategy` and
`ContainerAgitateStrategy`.

### The four gates

A planned action has to clear all four, in this order, or the result is `NONE` and nothing moves.

1. The active behaviour profile's `recovery_allowed_actions` must contain it.
2. `SceneRecoveryPolicy.enabled` must be true and the resolved mode must be in `apply_modes`.
3. A physical action must have a `FixtureEnvelope`, which is enforced at construction.
4. The anti-loop history and the per-action and per-attempt budgets must allow it.

`ContainerAgitateStrategy` carries a fifth gate of its own: it returns `NONE` until an operator sets
a non-zero `max_agitate_amplitude_mm` on the envelope. When armed, `execute_recovery_motion` drives
three bounded waypoints, re-clamping every one against the envelope and aborting on any
non-`EXECUTED` `MotionResult`. Two templates exist: an air-shake oscillation while
`agitate_contact_depth_mm` is at its `0.0` default, and above it a contact redistribute that
descends, sweeps along +X and retracts. No built-in profile lists `container_agitate`, so the outer
gate drops it unless an operator writes a profile that does.

### Failure class to action

`RecoveryDispatcher` is a pure mapping from `GraspFailureReason` to an action priority tuple, with
an operator-supplied `overrides` mapping that replaces the default sequence for any listed class.
Two entries in the default map are rules rather than tuning:

- A planner refusal (`MOTION_PLAN_REFUSED`) maps to `RESCAN` and nothing else. Re-planning the same
  pose achieves nothing, and every other action either commands motion or silently redirects the
  cell to an object the operator did not ask for. Falling through to one of those after the cell has
  just declined to move is how a refusal becomes a motion.
- `CONTROLLER_NOT_OPERATIONAL` maps to no action at all. A stopped cell needs a person, not a retry,
  and least of all a motion. `TARGET_LABEL_NOT_FOUND`, `TOPOLOGY_RISK_REJECTED`,
  `SEMANTIC_REJECTED` and `DEFORMABLE_ROUTING_REQUIRED` are empty for the same kind of reason.

### The loop

`run_recovery_loop` drives failed pick, plan, execute, retry pick. `AutonomousGraspService` calls it
from `_run_with_recovery`, which returns a plain single pick when the policy is disabled or when the
resolved mode is not in `apply_modes`. When the loop does run, the recovery trail is folded into the
report's telemetry.

## Traps

- The config key is `robot.grasping.recovery.enabled` for the orchestrator that wraps the whole
  attempt. `robot.grasping.dense_recovery` is a separate block with its own `enabled` switch, and
  turning one on does not turn the other on.
- `allowed_actions` defaults to an empty tuple in the config schema and to
  `(RESCAN, NEXT_VIEWPOINT)` on the runtime policy. Enabling recovery without naming actions gets
  you the config default, which is nothing.
- The `verification_heavy` preset sets `recovery.enabled: true` but leaves `apply_modes` at its
  default, which does not include `closed_loop`. Recovery there stays inert until a cell adds the
  mode to that list.

## See also

- [`grasping/`](../README.md) for the tier this closes the loop for
- [`loop/`](../loop/README.md) for the orchestrator that reports the failure this package reacts to
- [`types/`](../types/README.md) for `GraspFailureReason`, the vocabulary the dispatcher maps from
- [`closed_loop/`](../closed_loop/README.md) for the other second chance, taken before the failure
- [`safety/`](../../safety/README.md) for the guards every recovery waypoint still passes through
- [`docs/grasping-config-reference.md`](../../../../docs/grasping-config-reference.md) for the
  `recovery` and `dense_recovery` blocks and the modes they fire in
