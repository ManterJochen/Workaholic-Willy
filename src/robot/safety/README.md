# The safety gate every motion passes (`src/robot/safety`)

`SafetyPreflight` is the gate a driver asks before it commands a motion: six guards in a fixed order,
and the first one that rejects vetoes the move.

```python
from willy import SafetyPreflight, create_arm, load_tree

tree = load_tree()                                      # the cell WILLY_PROFILE names
gate = SafetyPreflight.from_tree(tree)
print(gate.guard_names)                                 # the guards that run, in order
print(gate.omitted_guards)                              # the families this cell switched off

arm = create_arm(tree.robot.vendor, config=tree.robot)  # read for its limits; nothing connects
refusal = gate.gate_planned_path(waypoints, arm=arm)    # joint waypoints; None when every sample passes
print("clear" if refusal is None else refusal)
```

The UR and other vendor drivers build this gate from the same tree, so a `Robot` or a `Cell` on one of
them already runs it before each move; the dummy arm carries none.
Quick checks: 
[gate_the_whole_path.py](../../../examples/offline/safety/gate_the_whole_path.py) and
[self_collision_backend.py](../../../examples/offline/safety/self_collision_backend.py).
[`scripts/checks/safety_guards.py`](../../../scripts/checks/safety_guards.py) makes every guard your cell
wires refuse a violation of its own family, and exits 1 when one does not.

## The six guards

| Order | Guard | Rejects when | Needs |
| --- | --- | --- | --- |
| 1 | `workspace` | the TCP target is outside the box, shrunk by `limits.workspace_margin_mm` on every face | a target pose in `Frame.BASE` |
| 2 | `joint_limit` | an axis is outside its limit, less `margin_deg` at both ends | target joints |
| 3 | `ik_quality` | the solution is non-finite, the wrong size, a large jump, near a limit, or near-singular | target joints; the arm for the singularity probe |
| 4 | `self_collision` | a link, the tool or a declared fixture comes closer than `min_distance_mm` (10), or a box a camera saw closer than `perceived_min_distance_mm` (5) | target joints and a kinematics model |
| 5 | `payload` | mass, centre of gravity or inertia is outside the declared envelope | config only |
| 6 | `motion_continuity` | the step from the last accepted target is larger than its cap | the previous accepted target |

The workspace box always runs. Every other guard runs only while its family's `enforce` key under
`robot.safety` is true: `enforce: false` removes the guard when the gate is built, and `omitted_guards`
names what is gone. Every verdict is a frozen `SafetyDecision` with a `SafetyReason` (`OK`, `WORKSPACE`,
`JOINT_LIMIT`, `IK_QUALITY`, `SELF_COLLISION`, `PAYLOAD`, `CONTINUITY`, `UNAVAILABLE`) that maps to one
`MotionStatus`, so a driver turns a rejection into a typed `MotionResult` without reclassifying it.

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `SafetyPreflight` | `from_tree(tree)`, `from_safety_config(safety, workspace)` | `evaluate(context)` | a `SafetyDecision` |
| `SafetyPreflight` | the same | `screen(context)`: `evaluate`'s verdict on a target nothing is sent to, the continuity memo left as it was | a `SafetyDecision` |
| `SafetyPreflight` | the same | `gate_joint_target`, `gate_joint_path`, `gate_planned_path` | `None` when clear, else the refused `MotionResult` |
| `SafetyPreflight` | the same | `exact_pairs(arm)`, `set_perceived_obstacles(boxes)` | the exact guard's pair rule and distances (`planning.band.ExactPairs`), or `None` without the exact backend; how many guards took the camera's boxes |
| `SafetyAttestation` | `SafetyAttestation.of(arm)`, `robot.safety()`, `cell.safety()` | `render()` | what an arm will refuse |
| `ContinuousCollisionMonitor` | `from_model(...)` with `ContinuousGuardProfile(enabled=True)` | `check(joints)` | a verdict per control step |

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `JointLimitTableMissing` from `create_arm` | the joint-limit guard is wired and no table resolves (KUKA, the sim) | set `robot.safety.joint_limits.min_deg` and `max_deg` |
| `ConfigError` when the gate is built | an exact-mesh guard reads hand geometry and `robot.gripper.model` names no hand | name the hand the flange carries |
| `ConfigError` when the gate is built | the declared tool frame places no hand model (a mirror, an oblique axis) | declare the tool frame the hand model admits |
| a config refused at load | `self_collision.perceived_min_distance_mm` plus `planning_world.perceived.margin_mm` falls short of `self_collision.min_distance_mm`, the step every path is sampled at | raise either of the first two until they reach the step |
| a config refused at load | `planning_world.perceived.voxel_size_mm` is coarser than 10 while `perceived_min_distance_mm` is under `min_distance_mm`: the seen boxes are fitted to one pixel per voxel, and that distance was measured at a 10 mm voxel | keep `voxel_size_mm` at 10 or finer |
| a config refused at load | `self_collision.perceived_min_distance_mm` is below 5: where only the camera's boxes refuse the planner's world, the exact guard alone decides them at that distance (the owner, 2026-10-01) | set it to 5 or more |
| a path gate refusal, `UNSUPPORTED` | no self-collision guard, the capsule proxy, no reach for the arm, or too many samples | `path_judge_refusal(arm)` names the cause before any motion |
| a UR connect refusal | `payload.enforce: true` with `mass_kg: 0.0`, or a mass with `cog_mm` at the origin | weigh the tool and measure its centre of gravity, or `enforce: false` for a bare flange |
| `UNAVAILABLE` on a move | a guard lacks the config, telemetry or asset it needs while its family is enforced | the message names what is missing |

## Self collision: exact meshes, or the capsule proxy

`self_collision.backend` ships as `fcl`: exact mesh distance with Coal, or with python-fcl where Coal is
absent.

The meshes ship in `data/`: an arm bundle `{model}_collision_meshes.npz` for `ur3`, `ur3e`, `ur5`,
`ur5e`, `ur10`, `ur10e` and `ur16e`, and a hand bundle `{hand}_hand_meshes.npz` for `robotiq_2f85`,
`robotiq_hande` and `schunk_egu50`, composed onto the arm when the guard loads. `bundles.json` records
each bundle's source and a hash of its arrays. With `mesh_dir: null` the guard loads its own model's
bundle from `data/`; `mesh_dir` names a directory of bundles you baked. A model with no bundle runs the
capsule proxy.

**What the exact guard judges.** Every pair of parts more than one DH frame apart, a wrist camera's parts
also against wrist_2: `MeshSelfCollisionBackend.checks(part_a, part_b)` is that rule asked of one pair, and
the rule `evaluate` skips pairs by; `part_frames` names each part's frame and `distance_mm(...)` the exact
distance of two parts at a configuration, for a sentence, never a verdict. `SelfCollisionGuard.exact_pairs`
and `SafetyPreflight.exact_pairs` hand that rule and those distances to the planner's band admission
(`planning/band.py`), and are `None` wherever the exact backend does not run: the capsule proxy decides
nothing the planner refused. There the exact guard decides the arm's own pairs among `upper_arm` through
`wrist_3`, the hand and the wrist cameras; the `shoulder_link`, which is also the planner's only model of
the base, and any pair padded beyond `planner_margin_mm` stay the planner's
([guide 04](../../../docs/guide/04-robot-and-safety.md), section 6). `SafetyPreflight.perceived_obstacles`
names the boxes a camera saw that this guard holds: where only those refuse the planner's world, no part
is carried and the hand reads empty and open, the band admission sets exactly them aside in the planner and
this guard decides them, judging every refused sample again at `perceived_min_distance_mm`, never less than
5 mm (a load refuses less).

**The camera's boxes, turned, at their own distance.** A declared fixture is an axis-aligned box. A box a
camera saw carries the turned box the planner holds (`AxisAlignedBox.turned`), and the exact backend judges
that one, turned about base Z; the capsule proxy judges its enclosure. The guard asks the arm and the
declared fixtures at `min_distance_mm` (10) first, then the seen boxes alone at
`self_collision.perceived_min_distance_mm` (5), each already grown by `planning_world.perceived.margin_mm`
(15), so the arm keeps 20 mm off a seen face, measured to the thinned cloud, one point per 10 mm voxel. A
pixel the thinning dropped was measured as near as 4.1 mm inside its box, about 9 mm from the arm at a
sample that passes; a measurement, not a bound ([guide 04](../../../docs/guide/04-robot-and-safety.md), 5.5).
A refusal's detail says where the box stood: `fixture` (`seen` or `declared`), `box_centre_mm`,
`box_size_mm`, `box_yaw_deg`, `box_corners_mm`, `box_note` and the `joints_deg`, and the path gates log it.

## How each guard decides

- **Workspace.** A margin that would invert an axis raises. A joint-only command carries no pose and
  passes here; the joint and self-collision guards judge it.
- **Joint limits.** `min_deg` and `max_deg` from config win; otherwise the built-in `UR_JOINT_LIMITS_DEG`
  table (plus or minus 360 degrees per axis, `ur3` to `ur20`); otherwise `UNAVAILABLE`. The table is the
  manufacturer envelope, not the installation limits set on the controller, which the controller
  enforces itself as a protective stop.
- **IK quality.** Limit proximity is skipped when no envelope resolves. The singularity probe runs only
  on an arm with `has_native_fk`, and a probe that fails is `UNAVAILABLE`.
- **Self collision.** The base column, a tool shape at the TCP, the arm links, the boxes under
  `self_collision.fixtures` and the boxes the cameras saw. Arm links need a UR arm or an explicit
  `kinematics_model`; without one only the base, the tool and the fixtures are checked. `tool_model: finger`
  models descending two-finger jaws.
- **Payload.** The UR driver pushes mass and centre of gravity to the controller at connect while
  `enforce` is true, and drops the connection if the push fails. KUKA payload is config only.
- **Motion continuity.** Caps the joint, TCP and orientation step against the last accepted target. The
  first command after `reset()` passes.
- **A planned move.** Both drivers skip `ik_quality` and `motion_continuity` on the planner's final
  configuration, because the planner's path replaces the jump they guard against. The other four run.

## Paths, not only end points

`gate_joint_path` and `gate_planned_path` judge every configuration of a path before any of it is
commanded, with the joint-limit, self-collision (fixtures included) and payload guards. No key turns
them off and there is no stride. The step is the self-collision margin (`path_step_mm`), and the reach
comes from the arm and what its flange carries: the hand, a wrist camera and a declared carried part.

`ContinuousCollisionMonitor` checks exact meshes at every control step of a move, arm against itself and
against fixtures. A check that overruns its budget or cannot run stops the move. It is opt-in: the
default `ContinuousGuardProfile` has `enabled=False`, a margin of 8.0 mm and a budget of 12.0 ms, and
only the Isaac driver runs one, when a sim runner such as `run_dense_pick` asks for it. Keep the margin
under 19.6 mm, the wrist clearance of a natural grasp, or a good pick stops.

**A real cell still needs the vendor's safety-rated stop,**
**an independent emergency-stop circuit, and compliance with ISO 10218, ISO/TS 15066 and ISO 13849.**

## Status

| Capability | Evidence |
| --- | --- |
| The six guards and the typed decision on every move | measured on a UR10 |
| Payload pushed to a UR controller at connect | measured on a UR10 |
| The continuous monitor | measured on a UR10 |
| Arm-against-arm self collision on a physical robot | measured on a UR10 |


## Files

| File | Holds |
| --- | --- |
| [`preflight.py`](preflight.py) | `SafetyPreflight`: the builders, `evaluate` and `screen`, the path gates |
| [`decision.py`](decision.py) | `SafetyDecision`, `SafetyReason` and the reason to `MotionStatus` table |
| [`guard.py`](guard.py) | the `SafetyGuard` Protocol and `SafetyContext` |
| [`workspace.py`](workspace.py), [`joint_limits.py`](joint_limits.py), [`ik_quality.py`](ik_quality.py) | the first three guards, and `JointLimitTableMissing` |
| [`self_collision.py`](self_collision.py), [`payload.py`](payload.py), [`continuity.py`](continuity.py) | the last three guards |
| [`path_samples.py`](path_samples.py) | a move turned into the configurations a gate judges |
| [`singularity.py`](singularity.py) | Jacobian and singular-value analysis |
| [`continuous_monitor.py`](continuous_monitor.py) | the per-step monitor and its profile |
| [`attestation.py`](attestation.py) | `SafetyAttestation` and `SafetyPosture` |
| `_capsule.py`, `_ur_kinematics.py`, `_fcl_self_collision.py` | capsule distances, the UR DH tables, the exact mesh backend |
| [`planning/`](planning/README.md) | the planner sidecar and the two external engines |
| `data/` | the committed mesh bundles and `bundles.json` |

## Details

- Guide: [robot and safety](../../../docs/guide/04-robot-and-safety.md), sections 4 and 5
- Maths: [safety-math.md](../../../docs/safety-math.md), distances, DH kinematics, singularity
- Runbook: [the first pick on a real cell](../../../docs/runbooks/real_cell_first_pick.md)
- Motion types: [`../core/`](../core/README.md) for `MotionCommand`, `MotionResult` and `MotionStatus`
- Tests: `tests/test_safety_preflight.py`, `tests/test_joint_path_gate.py`, `tests/test_safety_attestation.py`
