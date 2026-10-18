# The safety gate every motion passes (`src/robot/safety`)

`SafetyPreflight` is the gate a driver asks before it commands a motion: six guards in a fixed order,
and the first one that rejects vetoes the move.

```python
from willy import load_tree
from src.robot.drivers import create_arm
from src.robot.safety import SafetyPreflight

tree = load_tree()                                      # the cell WILLY_PROFILE names
gate = SafetyPreflight.from_tree(tree)
print(gate.guard_names)                                 # the six guards, in order: every cell runs all of them

arm = create_arm(tree.robot.vendor, config=tree.robot)  # read for its limits; nothing connects
refusal = gate.gate_planned_path(waypoints, arm=arm)    # joint waypoints; None when every sample passes
print("clear" if refusal is None else refusal)
```

The UR and other vendor drivers build this gate from the same tree, so a `Robot` or a `Cell` on one of
them already runs it before each move; the dummy arm carries none.
[`scripts/checks/safety_guards.py`](../../../scripts/checks/safety_guards.py) makes every guard of your cell
refuse a violation of its own family, and exits 1 when one does not or is missing.

## The six guards

| Order | Guard | Rejects when | Needs |
| --- | --- | --- | --- |
| 1 | `workspace` | the TCP target is outside the box, shrunk by 20 mm on every face (`workspace.WORKSPACE_MARGIN_MM`) | a target pose in `Frame.BASE` |
| 2 | `joint_limit` | an axis is outside the arm's own limit, less 5 degrees at both ends (`JOINT_LIMIT_MARGIN_DEG`), or more than half a turn from home on a real arm | target joints; the home the arm states |
| 3 | `ik_quality` | the solution is non-finite, the wrong size, a jump over 1.0 rad, within 2 degrees of a limit, or near-singular (a least singular value under 0.005, a condition number over 250) | target joints; the arm for the singularity probe |
| 4 | `self_collision` | a link, the tool or a declared fixture comes closer than `min_distance_mm` (3), or a box a camera saw closer than `perceived_min_distance_mm` (2) | target joints and a kinematics model |
| 5 | `payload` | mass, centre of gravity or inertia is outside the declared envelope | config only |
| 6 | `motion_continuity` | the step from the last accepted target is over 45 degrees of a joint, 30 of the TCP's turn or 250 mm | the previous accepted target |

Every guard runs on every cell: no key switches one off since 2026-10-17 (the owner's cleanup took the `enforce`
keys and fixed the numbers above at the owner's cell's). They are the numbers of the arm the gate is built for
([`guard_numbers.py`](guard_numbers.py)): every real arm's are the cell's, and the Isaac arm states its own in code
([`src/willy_sim/guard_numbers.py`](../../willy_sim/guard_numbers.py): a half-turned base, its whole joint envelope,
IK and continuity limits sized for a position-controlled drive). Every verdict is a frozen `SafetyDecision` with a
`SafetyReason` (`OK`, `WORKSPACE`, `JOINT_LIMIT`, `IK_QUALITY`, `SELF_COLLISION`, `PAYLOAD`, `CONTINUITY`,
`UNAVAILABLE`) that maps to one `MotionStatus`, so a driver turns a rejection into a typed `MotionResult` without
reclassifying it.

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
| `JointLimitTableMissing` from `create_arm` | no table resolves for the arm (no description), the arm states no home, or the window about home leaves an axis no range (a home more than half a turn outside the table) | describe the arm in `config/arms/<model>.yaml`; set `robot.motion.home_deg`, in degrees as the pendant shows the joints |
| `ConfigError` when the gate is built | an exact-mesh guard reads hand geometry and `robot.gripper.model` names no hand | name the hand the flange carries |
| `ConfigError` when the gate is built | the declared tool frame places no hand model (a mirror, an oblique axis) | declare the tool frame the hand model admits |
| a config refused at load | `self_collision.perceived_min_distance_mm` plus the 6 mm every camera world grows its boxes by falls short of `self_collision.min_distance_mm`, the step every path is sampled at | raise the first, or lower the step |
| a config refused at load | `self_collision.perceived_min_distance_mm` is below 1: where only the camera's boxes refuse the planner's world, the exact guard alone decides them at that distance (the owner, 2026-10-12) | set it to 1 or more |
| a path gate refusal, `UNSUPPORTED` | no self-collision guard, the capsule proxy, no reach for the arm, or too many samples | `path_judge_refusal(arm)` names the cause before any motion |
| a UR connect refusal | `payload.mass_kg: 0.0`, the shipped "not weighed yet", or a mass with `cog_mm` at the origin | weigh what the flange carries and measure its centre of gravity |
| `UNAVAILABLE` on a move | a guard lacks the config, telemetry or asset it needs | the message names what is missing |

## Self collision: exact meshes, or the capsule proxy

`self_collision.backend` ships as `fcl`: exact mesh distance with Coal, or with python-fcl where Coal is
absent.

The meshes ship in `data/`: an arm bundle `{model}_collision_meshes.npz` for `ur3`, `ur3e`, `ur5`,
`ur5e`, `ur10`, `ur10e` and `ur16e`, and a hand bundle `{hand}_hand_meshes.npz` for `robotiq_2f85`,
`robotiq_hande` and `schunk_egu50`, composed onto the arm when the guard loads. Every arm bundle is
baked in its links' DH frames from Universal Robots' own collision STL files, through the rendered
description, by `scripts/arms/bake.py`. `bundles.json` records each bundle's
source and a hash of its arrays. With `mesh_dir: null` the guard loads its own model's bundle from
`data/`; `mesh_dir` names a directory of bundles you baked. A model with no bundle runs the capsule
proxy.

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
names the boxes a camera saw that this guard holds: where only those refuse the planner's world, whatever the
hand holds (no cell models a part in the jaws since 2026-10-17), the band admission sets exactly them aside in the
planner and this guard decides them, judging every refused sample again at `perceived_min_distance_mm`, never less
than 1 mm (a load refuses less).

**Where it stands the hand.** The links stand on the model's nominal DH table; the flange, and with it the hand, its
fingers and a wrist camera, stands where the arm's controller puts it, where the arm says
(`core.arm_capabilities.KnowsItsFlange`, `_ur_kinematics.arm_link_transforms_mm`; the owner, 2026-10-12). A
controller calibrated at the factory puts the flange millimetres from the table's (6.0 to 6.3 mm and 0.6 degrees on
the owner's UR10), and the camera's boxes stand in its world, placed by the TCP it reports. The UR driver says it on
its controller's own rows once they put the flange where the controller says it stands at connect, within 0.1 mm and
0.01 degrees (`URRobotArm.controller_flanges_mm`); every other arm, and a UR whose rows cannot be read or are not the
ones it computes on, has its flange on the table, as before. One configuration judged alone is placed bit for bit as
in a path judged whole.

**The camera's boxes, turned, at their own distance.** A declared fixture is an axis-aligned box. A box a
camera saw carries the turned box the planner holds (`AxisAlignedBox.turned`), and the exact backend judges
that one, turned about base Z; the capsule proxy judges its enclosure. The guard asks the arm and the
declared fixtures at `min_distance_mm` (3) first, then the seen boxes alone at
`self_collision.perceived_min_distance_mm` (2), each already grown by the 6 mm every camera world grows its
boxes by (`perceived.CAMERA_BOX_MARGIN_MM`), so the arm keeps 8 mm off the thinned cloud, one point per 10 mm
voxel. A pixel the thinning dropped was measured as near as 4.1 mm inside a box grown by 15; a measurement, not
a bound ([guide 04](../../../docs/guide/04-robot-and-safety.md), 5.5).
A refusal's detail says where the box stood: `fixture` (`seen` or `declared`), `box_centre_mm`,
`box_size_mm`, `box_yaw_deg`, `box_corners_mm`, `box_note` and the `joints_deg`, and the path gates log it.

## How each guard decides

- **Workspace.** A margin that would invert an axis raises. A joint-only command carries no pose and
  passes here; the joint and self-collision guards judge it.
- **Joint limits.** The envelope the arm registry describes for the model the arm reports,
  `capabilities.model` (`config/arms/<model>.yaml`, plus or minus 360 degrees per axis, `ur3` to `ur20`),
  whatever its vendor; otherwise `UNAVAILABLE`, and `create_arm` refuses the arm. The Isaac arm reports the UR
  it drives and resolves that UR's envelope. On every real arm every full-turn joint (a UR's elbow excepted)
  stays within half a turn of the home the arm drives to, so a cable along the arm is never wound further; the
  Isaac arm keeps its whole envelope. That envelope is the manufacturer's, not the installation limits set on the
  controller, which the controller enforces itself as a protective stop.
- **IK quality.** Limit proximity is skipped when no envelope resolves. The singularity probe runs only
  on an arm with `has_native_fk`, and a probe that fails is `UNAVAILABLE`.
- **Self collision.** On the capsule proxy: the base column up to the shoulder joint the arm's description
  places, 80 mm round; the hand's fingers as one capsule along the grasp's closing axis at the TCP, from the
  hand's registry file (the 2F-85 every arm bundle carries where no hand is named); the arm links, 60 mm round;
  the boxes under `self_collision.fixtures` and the boxes the cameras saw. Arm links and the base need an arm
  whose `capabilities.model` the arm registry describes as a UR (`described_ur_model`), whatever its vendor: a
  real UR reports `ur.model`, the Isaac arm `sim.robot_model`. Any other arm gets only the tool and the
  fixtures. Until P0d D5 the Isaac arm reported `isaac-sim` and needed `self_collision.kinematics_model` to opt
  in; until 2026-10-17 the capsule's dimensions and the Isaac base's half turn were keys too. A tree that still
  writes one is refused at load.
- **Payload.** The UR driver pushes mass and centre of gravity to the controller at connect on every cell, and
  drops the connection if the push fails.
- **Motion continuity.** Caps the joint, TCP and orientation step against the last accepted target. The
  first command after `reset()` passes.
- **A planned move.** Both drivers skip `ik_quality` and `motion_continuity` on the planner's final
  configuration, because the planner's path replaces the jump they guard against. The other four run.

## Paths, not only end points

`gate_joint_path` and `gate_planned_path` judge every configuration of a path before any of it is
commanded, with the joint-limit, self-collision (fixtures included) and payload guards. No key turns
them off and there is no stride. The step is the self-collision margin (`path_step_mm`), and the reach
comes from the arm and what its flange carries: the hand and a wrist camera.

**A path judged whole**, on every cell since 2026-10-17 (the owner's `self_collision.whole_path_judge`, on at the
cell). The gate first asks each guard of a joint move for the first sample it might refuse, over the whole path at once
(`first_suspect`): the joint limits on every sample at once, the payload once, the exact meshes with every part placed
at every sample in one pass (`_ur_kinematics.ur_link_transforms_mm_many`, bit for bit the chain one sample is placed
with). A sample before that one is passed only on a proof that every distance the guard would measure there
keeps its limit: the bounding spheres the guard culls by; a part's convex hull (Coal only), which holds the
mesh and is taken 0.001 mm nearer than it measures; or a distance measured at an earlier sample less how far
the pair can have moved since, an arm pair in either part's frame and a part against a box as the guard's own
box rule. Where none proves it, the pair is measured exactly as the guard measures it, and a distance under
the limit plus 1e-6 mm ends the pass there. From that sample on the gate judges every sample one at a time,
so the verdict, the sample it names and its message are those of judging every sample in turn
(`tests/test_a_whole_path_is_judged_as_every_sample_is.py`: routes, lines and paths over the cell's worlds
of 2026-10-07, the borderline at 3 mm give or take 1e-7 mm, the guard order). A guard that offers no such
pass, a site-local one, or the self-collision guard where it would answer with the capsule proxy, leaves
every sample to the loop, and the log says so once. On the desk the owner's route of 905 samples over 128
boxes took 12 ms instead of 0.96 s, the way home 9 to 11 ms instead of 1.2 to 1.4 s, a line down 4 to 6 ms
instead of 160 to 180 ms. The UR driver's mesh first asks the support plane and the robot's base of a whole
path the same way (`ExactPairs.first_low`, `first_near_base`), and takes its sentence from the configuration
the loop names.

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
| [`guard_numbers.py`](guard_numbers.py) | `GuardNumbers`, what the guards hold an arm to that no cell configures, and `CELL`, every real arm's |
| `_capsule.py`, `_ur_kinematics.py`, `_fcl_self_collision.py` | capsule distances, the UR DH tables and the chain of many configurations at once, the exact mesh backend and its whole-path pass |
| `_ur_ik.py` | the closed-form inverse kinematics of the UR flange and its branches; on a controller's own calibrated rows (`_ur_kinematics.URDhChain`), the configuration nearest a seed as its `getInverseKinematics` answers it, or why that cannot be vouched for (`ur_chain_ik_nearest`; every line a cuRobo UR judges is solved so since 2026-10-17) |
| [`partner_arm.py`](partner_arm.py) | another arm of the cell as the boxes this arm keeps its distance from (`partner_world`, `PartnerWorld`, `partner_cuboids`, `hull_gap_mm`) |
| [`planning/`](planning/README.md) | the planner sidecar and the two external engines |
| `data/` | the committed mesh bundles and `bundles.json` |

## Details

- Guide: [robot and safety](../../../docs/guide/04-robot-and-safety.md), sections 4 and 5
- Maths: [safety-math.md](../../../docs/safety-math.md), distances, DH kinematics, singularity
- Runbook: [the first pick on a real cell](../../../docs/runbooks/real_cell_first_pick.md)
- Motion types: [`../core/`](../core/README.md) for `MotionCommand`, `MotionResult` and `MotionStatus`
- Tests: `tests/test_safety_preflight.py`, `tests/test_joint_path_gate.py`, `tests/test_safety_attestation.py`
