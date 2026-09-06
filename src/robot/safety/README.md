# robot.safety

The fail-closed gate every commanded motion passes through before it runs. If a check cannot be
made, the move is refused rather than quietly allowed.

## What this package guarantees

`SafetyPreflight` runs a fixed-order chain of guards over a commanded move. The first guard that
rejects vetoes it. A guard that cannot decide, because config, telemetry or an asset is missing,
returns `UNAVAILABLE`, and while its family is enforced that also refuses the move.

Safety is the top of the runtime authority hierarchy. It outranks the deterministic geometry
pipeline, the recovery logic and the optional learned layer. Anything above it may reorder or filter
candidates the guards have already cleared; nothing above it can overturn a rejection.

The emergency stop is deliberately not here. It is a hardware and controller function, and nothing
in this package can enable, disable, route or observe it.

## The chain

Six guards, in this order:

| Order | Guard | Rejects when | Needs |
| --- | --- | --- | --- |
| 1 | `workspace` | the TCP target is outside the Cartesian box, after the margin is subtracted | `target_pose` |
| 2 | `joint_limit` | any axis is outside its hard limit, after the margin is subtracted | `target_joints` |
| 3 | `ik_quality` | the IK solution is non-finite, the wrong DoF, a large jump, close to a limit, or near-singular | `target_joints`, and `arm` for the forward-kinematics probe |
| 4 | `self_collision` | a link, tool or fixture pair is closer than `min_distance_mm` | `target_joints` and a kinematics source |
| 5 | `payload` | mass, centre of gravity or inertia is outside the declared envelope | config only |
| 6 | `motion_continuity` | the step from the last accepted target is larger than the cap | the previous accepted target |

The workspace box is the one guard that is always wired. Every other family is appended by
`from_safety_config` only when its `enforce` flag is set. Setting `enforce: false` removes the guard
from the pipeline at construction: its safety surface is gone, not merely quiet, until the flag comes
back. `SafetyPreflight.omitted_guards` names what was left out, so a running cell can be asked.

Every verdict is a frozen `SafetyDecision` carrying a closed-set `SafetyReason` (`OK`, `WORKSPACE`,
`JOINT_LIMIT`, `IK_QUALITY`, `SELF_COLLISION`, `PAYLOAD`, `CONTINUITY`, `UNAVAILABLE`), a message for
a person, and a structured detail payload. A non-OK reason maps deterministically to a
`MotionStatus`, so the driver boundary turns a rejection into a typed `MotionResult` without
reclassifying anything.

## Contents

| File or directory | Contents |
| --- | --- |
| `preflight.py` | `SafetyPreflight`, the `WorkspaceSafetyGuard` adapter, and the workspace margin shrink |
| `decision.py` | `SafetyDecision`, `SafetyReason`, and the reason to `MotionStatus` table |
| `guard.py` | the `SafetyGuard` Protocol and `SafetyContext` |
| `workspace.py` | `WorkspaceGuard`, the box plus pose diversity, and the only stateful guard |
| `joint_limits.py` | `JointLimitGuard`, `resolve_joint_limits_deg`, and the `UR_JOINT_LIMITS_DEG` table |
| `ik_quality.py` | `IKQualityGuard` |
| `self_collision.py` | `SelfCollisionGuard`, with the capsule and exact-mesh backends |
| `payload.py` | `PayloadGuard` |
| `continuity.py` | `MotionContinuityGuard` |
| `singularity.py` | Jacobian and singular-value analysis helpers, and `SingularityGuard` |
| `continuous_monitor.py` | the opt-in per-control-step collision-avoidance monitor |
| `attestation.py` | `SafetyAttestation`, `SafetyPosture` and `SafetyGated`: ask an arm what it enforces |
| `_capsule.py` | `Capsule`, `AxisAlignedBox` and the closed-form signed-distance helpers |
| `_ur_kinematics.py` | the UR DH tables, `ur_link_origins_mm` and `ur_link_transforms_mm` |
| `_fcl_self_collision.py` | the exact mesh-against-mesh backend |
| [`planning/`](planning/README.md) | the trajectory-planner sidecar and the anchor for both external engines |
| `data/` | the committed collision-mesh bundles |

## The collision-mesh bundles ship

`data/` holds three committed bundles: `ur5e_collision_meshes.npz`, `ur3e_collision_meshes.npz` and
`schunk_egu50_collision_meshes.npz`, the last being the ur5e arm with a different gripper. They are
per-DH-frame vertex and face arrays, so no config is needed to find them: with
`self_collision.mesh_dir` left at `null` the guard loads the bundle for its own model out of this
directory. `mesh_dir` names an alternate directory for a bundle you baked yourself.

A model gets exact meshes as soon as `{model}_collision_meshes.npz` lands beside these, with no code
change. `collision_mesh_variant` selects a per-gripper bundle instead of the model default, for a
cell running an end-effector other than the baked Robotiq 2F-85.

## The exact-mesh guard falls back rather than failing closed

`self_collision.backend` ships as `fcl`, and `python-fcl` is in `requirements.txt`, so an engine is
part of the default install. Coal is preferred over python-fcl where it is present; the swap is
behaviour-identical, because both run the same meshes, the same pairs, the same thresholds and the
same distance query. `scripts/ext_deps/install.ps1` installs Coal, and the resolution happens in one
place, `planning/environment.py`.

If neither engine nor the bundle is available, the guard logs one warning naming the reason and then
runs the capsule path. It does not refuse the motion. This matters twice over:

- The `SelfCollisionSafetyConfig` docstring in `src/config/schema/robot/safety_schema.py` says that
  `fcl` without a populated `mesh_dir` reports `UNAVAILABLE` and refuses while `enforce` is true.
  The code does not do that. Treat the code as authoritative and the docstring as stale.
- The capsule proxy is materially coarser. Its default 60 mm link radius already over-rejects
  legitimate reach-down grasps, and it skips the wrist pairs the exact backend catches. A cell that
  silently swapped one for the other would be a safety-relevant regression, which is why the
  fallback is logged with a reason token rather than being invisible.

Verify what actually resolved on a box with `python -m src.robot.safety.planning --check`.

## Usage

There is no command-line entry point for the guards themselves; they are a library consumed by
drivers. The `planning/` subpackage does have one.

```python
from src.robot.safety import SafetyPreflight

# Built once per driver, normally with
# SafetyPreflight.from_safety_config(safety_cfg, workspace_cfg).
ctx = self._preflight.context_for_pose(pose, current_joints=self.get_joint_positions(), arm=self)
decision = self._preflight.evaluate(ctx)
if (result := SafetyPreflight.as_motion_result(decision, command, target_pose=pose)) is not None:
    return result  # the rejection, already a typed MotionResult
```

`SafetyPreflight(guards)` is the ordered pipeline. Build it with
`from_safety_config(safety_cfg, workspace_cfg, *, extra_guards=())` or `from_workspace_only(...)`.
Make a context with `context_for_pose(...)` or `context_for_joints(...)`; evaluate with
`evaluate(ctx)`; gate a deliberate joint move with `gate_joint_target(joints, *, arm=None)`; gate a
whole planned path with `gate_trajectory(waypoints, *, arm=None, stride=None)`; clear the memo with
`reset()`; translate a rejection with the static `as_motion_result(...)`. Introspect through
`guards`, `guard_names`, `omitted_guards` and `checks_trajectories`.

`SafetyDecision` is built with `accept(guard)`, `reject(guard, reason, ...)` or
`unavailable(guard, ...)` and read through `accepted`, `rejected` and `motion_status`.

To add a guard: implement `evaluate(ctx) -> SafetyDecision`, give it a stable `name`, place that name
in `SafetyPreflight._CANONICAL_ORDER` or inject the instance through `extra_guards`, and wire an
`enforce`-gated append in `from_safety_config`.

## Per-guard behaviour worth knowing

Workspace. The configured box is shrunk by `workspace_margin_mm` on all six faces; a margin that
would invert an axis raises. A joint-only command with no `target_pose` accepts here and relies on
the joint and self-collision guards. The pose-diversity check is a calibration concern and is not
consulted per move.

Joint limits. The envelope resolves in this order: explicit `min_deg` and `max_deg` from config; then
the built-in `UR_JOINT_LIMITS_DEG` table, which is the manufacturer envelope of plus or minus 360
degrees per axis for `ur3`, `ur3e`, `ur5`, `ur5e`, `ur10`, `ur10e`, `ur16e` and `ur20`; otherwise
`UNAVAILABLE`. `margin_deg` comes off both ends. KUKA and the sim have no built-in table, so a cell
either supplies limits in YAML or the guard fails closed.

IK quality. On a pre-resolved `target_joints`: non-finite rejects; a wrong DoF rejects and needs
`arm`; a per-axis jump over `max_jump_rad` rejects; proximity closer than `limit_proximity_deg` to a
resolved limit rejects, and is skipped when no envelope resolves; the singularity probe runs only
when the arm advertises `has_native_fk`, and a probe failure is `UNAVAILABLE`.

Self-collision. The registry is the base column, a tool capsule rooted at the TCP, the per-link arm
capsules and the declared fixture boxes. Arm-link capsules need `target_joints` and either a genuine
`vendor == "ur"` arm or an explicit `kinematics_model`, which is what lets a non-UR vendor that is
physically a UR opt in. Without one of those, only base, tool and fixtures are checked.
`kinematics_base_yaw_deg` reconciles the bundled DH base frame with the base frame poses and fixtures
are expressed in; the default `0.0` changes nothing, and because arm-against-arm distance is
rotation-invariant this knob cannot alter it. `tool_model` picks the tool shape: `capsule`, the
default, is a rotation-invariant cylinder along the approach axis, and `finger` is a thin capsule
along the closing axis, which is the faithful footprint of descending two-finger jaws.

Payload. Validates that mass is between zero and `max_mass_kg` and that inertia is non-negative. This
is defence in depth, since the schema enforces the same bounds. The UR driver additionally pushes
mass and centre of gravity to the controller on `connect()` while `enforce` is true, and rolls the
connection back if that push fails, because an unverified payload on a connected controller is worse
than no connection. Two payload configurations refuse to connect at all, before the socket opens:
`enforce: true` with `mass_kg: 0.0`, which would overwrite the controller's own payload for a
mounted tool with zero, and a positive `mass_kg` with `cog_mm` left at the origin, which would
declare that tool a point mass at the flange face. A genuinely bare flange is expressed by
`enforce: false`, which never touches the controller payload.

Motion continuity. Caps the joint step in degrees, the TCP step in millimetres and the orientation
step in degrees, measured against the previous accepted target memoised in the preflight. The first
command after `reset()` always passes. A DoF mismatch skips the joint-step check rather than
rejecting silently.

The planner path. `evaluate()` takes a `skip_guards` set, and its only caller is the cuRobo driver
path, which skips the two continuity checks (the `ik_quality` joint jump and the
`motion_continuity` step size) on the planned final configuration. Those two stand in for the
assumption that a blind interpolator will not teleport, which the planner replaces with a
continuous, collision-checked path. Workspace, joint limits, self-collision and payload still run on
that final configuration, so the exact-mesh gate is never skipped.

## Two things that gate a path rather than a point

`safety.trajectory_check` gates every configuration of a planned path before any of it is commanded,
using the same guards as a joint move: joint limits, self-collision including the declared fixtures,
and payload. It is off by default. While it is off, nothing examines the middle of a plan: the sim
applies each waypoint to the articulation and a real UR runs them in turn, so a path that grazes a
fixture halfway and lands clear passes every check there is. `stride` samples the path instead of
checking it, trading coverage for time, and the final configuration is checked whatever the stride.

`ContinuousCollisionMonitor` runs the exact-mesh backend over every interpolation waypoint of a move,
arm against itself and arm against fixtures, with a clearance margin that stops before contact and a
fail-safe of its own: a check that overruns its budget or cannot run returns a stop or a hold, never
a continuation without a result. It is opt-in behind `ContinuousGuardProfile`, whose default
`enabled=False` installs nothing. The default margin is 8.0 mm and the default budget is 12.0 ms,
roughly a 60 to 80 Hz monitor. The margin has a real ceiling: a natural grasp puts the arm's own
`wrist_1` and `wrist_3` pair at about 19.6 mm, so a margin above that false-stops a good pick.

Both of these reduce collision risk in software. Neither is a certified functional-safety stop. A
real cell still needs the vendor safety-rated stop, an independent emergency-stop circuit, and
compliance with ISO 10218, ISO/TS 15066 and ISO 13849.

## Asking an arm what it enforces

`SafetyPreflight` lives inside the driver, and a driver may carry none: the dummy driver does not, so
a run that gated nothing looks exactly like a run that passed six guards. `SafetyAttestation.of(arm)`
answers what a given arm object will actually refuse, and it travels in the run report. An arm that
does not answer is not read as safe: a caller's own `RobotArm` implementation gets
`SafetyPosture.UNSTATED` with `enforced` set to `False`.

It attests and does not enforce. Nothing in that module can stop a motion; only the preflight inside
the driver's own `move` can.

## Known limits

Arm-against-arm self-collision is UR-only. It comes from the bundled UR DH tables, so KUKA and the
sim get tool-against-base and tool-against-fixture coverage only.

The built-in joint-limit table is the manufacturer outer envelope, not the tuned installation limits
an operator configures on the controller. The controller enforces those itself as a protective stop,
and the UR interface does not expose them for reading.

KUKA payload writes are config-only. The vendor interface exposes no documented runtime payload call,
so the guard validates the envelope while the controller-side load stays whatever was configured
there.

The six-guard pipeline and the typed decision contract run live in the simulator picks and against
simulated UR controller software. No pick in this repository has been validated against a physical
robot, and the arm-against-arm self-collision path in particular has never run on one. Several
schema defaults are deliberately unusable markers rather than plausible guesses, a zero payload mass
and a centre of gravity at the origin among them, so a cell that has not been measured refuses
loudly instead of proceeding on a number nobody took.

Units are millimetres and XYZW quaternions, and everything is frame-tagged. `WorkspaceGuard` requires
`Frame.BASE`.

## See also

- [`planning/`](planning/README.md) for the trajectory planner and the two external engines
- [`../core/`](../core/README.md) for `MotionCommand`, `MotionResult` and `MotionStatus`
- `docs/safety-math.md` for the distance geometry, DH kinematics and singularity analysis
- `docs/runbooks/real_cell_first_pick.md` for bringing a real cell up through these guards
