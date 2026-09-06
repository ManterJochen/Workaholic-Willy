# 4. Robot, drivers and the safety stack

You have a config tree that validates ([01](01-configuration.md)) and need to know what stands
between that YAML and a moving arm: which driver gets built, which end-effector, which guards run in
which order, and which two external engines the motion stack needs before any number it produces
means anything.

The mathematics is not here. Capsule geometry, the DH chain, the base-yaw reconcile, the Jacobian
singularity test, exact mesh distance and the fail-closed authority rule live in
[docs/safety-math.md](../safety-math.md). Choosing and driving an end-effector is [06](06-grippers.md).

**Two facts to start from.** The default grasp attempt is open-loop: the decision gate, closed-loop
refinement and verification, recovery, fusion with its commit gate, the learned ranker and the
learned success model all ship `enabled: false`, and the simulation runners turn them on per
command-line flag in runner code rather than through config. The safety layer is the exception: the
`SafetyPreflight` is built by the driver, or injected into it, so it is wired on every live path
(section 4.1).

**Prerequisites.** A virtual environment with `requirements.txt`; `python` means that interpreter
throughout, and `python -m src.config` exits 0. Everything runs in the plain environment except the
mesh-bundle bake (5.2) and any on-box simulation run, which need the simulator's own interpreter.

Sibling guides: [01](01-configuration.md) . [02](02-models.md) . [03](03-calibration.md) . **04** .
[05](05-pick-loop.md) . [06](06-grippers.md)

---

## 1. The vendor contract

### 1.1 `RobotArm`

[`src/robot/core/robot_arm.py`](../../src/robot/core/robot_arm.py) defines a `@runtime_checkable` Protocol.
Every driver implements it; nothing above the driver layer imports a driver class.

| Group | Members | Contract |
|---|---|---|
| introspection | `capabilities`, `is_connected` | vendor, model, DoF, feature flags |
| lifecycle | `connect`, `disconnect` | idempotent both ways |
| state | `get_tcp_pose`, `get_joint_positions` | pose tagged `Frame.BASE`; joints in radians |
| motion | `move_joint`, `move_linear`, `stop` | raise `RobotMotionRejected` or `RobotConnectionError` |
| motion | `move_to_joints`, `move` | return a typed `MotionResult` |
| kinematics | `fk`, `ik` | BASE frame both ways |
| bool shims | `is_inside_workspace`, `move_to`, `move_home`, `wait_until_steady` | return `False` instead of raising |

Units are non-negotiable here: millimetres, XYZW quaternions, frame-tagged poses, joints in radians.
Vendor conventions stay inside the driver package, so UR's axis-angle rotation vectors and KUKA's
E6POS with ABC Euler degrees never escape it.

Prefer `move()` over `move_to()`. The bool pair collapses every failure into one `False`, so an
orchestrator cannot tell a workspace rejection from an IK failure from a controller refusal, while
`move()` returns a `MotionResult` carrying a `MotionStatus` from a closed set, six of whose values
are safety rejections, one per guard:

```powershell
python -c "from src.robot.core import MotionStatus; print([m.value for m in MotionStatus])"
```

### 1.2 The three optional capability Protocols

Some things only some controllers can do. Rather than bloat the contract-locked `RobotArm` surface
with methods most drivers would stub, they live as separate `runtime_checkable` Protocols in
[`src/robot/core/arm_capabilities.py`](../../src/robot/core/arm_capabilities.py), and callers feature-check
with `isinstance()`.

| Protocol | What it adds |
|---|---|
| `SupportsDigitalIO` | `set_digital_output`, `get_digital_input`, analog out, banked by `DigitalIOPort` (`standard`, `configurable`, `tool`) |
| `SupportsForceTorque` | `get_tcp_wrench` in N and Nm, BASE frame, plus `get_joint_torques` |
| `SupportsRobotStatus` | `get_robot_status` and `recover_from_protective_stop` |

The vendor-neutral `RobotMode` and `SafetyMode` enumerations exist so a controller's integer status
codes become portable words; UR maps its integers into them in `src/robot/drivers/ur/arm.py`. `URRobotArm` is the only
driver that implements any of the three, which an `issubclass` check confirms.
`SupportsDigitalIO` is load-bearing rather than decorative: it is what the two digital-I/O
end-effectors run on, and `from_robot_config` refuses to build either of them on an arm that does not
advertise it.

### 1.3 `Gripper` and `ObjectDetectingGripper`

[`src/robot/core/gripper.py`](../../src/robot/core/gripper.py) defines `is_connected`, `min_width_mm`,
`max_width_mm`, `connect`, `disconnect`, `activate`,
`set_width_mm(width_mm, *, speed=None, force=None)` and `get_width_mm()`. Two contract points bite
people. Widths are millimetres, but `force` and `speed` are normalised to `[0.0, 1.0]` and are
explicitly not newtons and not mm/s; a driver maps them to its own units inside itself, and the
OnRobot driver is the exception that takes force in newtons natively through its own config key. And
`ObjectDetectingGripper` is a capability extension with one method, `is_object_detected()`. A driver
that implements it opts the cell into post-close verification; a driver that does not is simply
trusted after the close command. That is the documented default, not a bug. Which drivers answer, and
with how much evidence, is in [`src/robot/grippers/README.md`](../../src/robot/grippers/README.md).

### 1.4 How vendor neutrality is enforced

The invariant is **lazy**, not **located**: no vendor SDK is imported at module import time. The
registry stores factory callables, not classes, with the driver import inside the factory body, so
importing `src.robot.drivers` on a machine with no `ur_rtde` does not fail. `create_arm`
structurally validates its result against the `RobotArm` Protocol, so a buggy factory cannot quietly
return the wrong shape. UR defers further still: `rtde_control` and `rtde_receive` are imported inside
`try`/`except ImportError` in `src/robot/drivers/ur/connection.py` with the hard failure deferred to `connect()`, while
`rtde_io` and the dashboard client are guarded independently so I/O and safety status can degrade
without killing the move core.

---

## 2. The drivers

### 2.1 What is registered, and what it is worth

```powershell
python -c "from src.robot.drivers import available_vendors; print(available_vendors())"
```

| Vendor | Registered | Needs SDK | Honest status |
|---|---|---|---|
| `ur` | yes | `rtde_control`, `rtde_receive` | measured against URSim, which runs the real UR controller software and speaks RTDE: dashboard power-on, `moveJ`, RTDE receive, digital outputs read back changed, tool-frame verification against the controller's own register, `setPayload` with rollback, and a genuine protective stop. No physical UR has ever executed a line of it. |
| `kuka` | yes | none (EKI and KRL over TCP and XML) | no motion has ever run against a live KRC4 or KRC5 from this codebase. The controller-side KRL program under [`config/robot/templates/kuka/`](../../config/robot/templates/kuka/) must be deployed by an integrator first. |
| `sim` | yes | `isaacsim` | the validation platform: the only path that has run full motion. |
| `dummy` | yes | none | intentionally trivial, and with **no kinematics**: `fk` returns the last recorded pose, `ik` the last recorded joints. |
| `franka`, `ros2` | **no** | | empty slots; `create_arm` raises `RobotConnectionError`. |

Details and bring-up checklists per vendor: [`src/robot/drivers/README.md`](../../src/robot/drivers/README.md)
and the per-vendor READMEs beside it.

### 2.2 Probe the host with the driver doctor

`python -m src.robot.drivers.doctor` prints one row per arm and gripper vendor: registered, SDK
importable, ready. It uses `importlib.util.find_spec` only and never imports an SDK. The plain table
exits 0, `--require <vendor>` exits 1 when that vendor is not ready and 2 on an unknown vendor string,
and `--json` emits the same records machine-readably. Run it and read your own machine rather than
trusting a pasted transcript.

The `registered` column is derived from the registries themselves, so a vendor with a driver can
never be reported as a reserved slot. Note the import path when checking by hand:
`src.robot.drivers` exports the arm-side `available_vendors`, and the gripper-side
`available_gripper_vendors` lives in `src.robot.grippers`.

The same readiness gate is wired into the config-driven boot path. `RuntimePickService`, which
`AutonomousGraspService.from_robot_config` builds, is the only caller of `require_arm_vendor_ready`,
so a misconfigured host fails there rather than deep inside `connect()`. Mock mode and the dummy
vendor are exempt so the offline test suite is never gated. The simulator cell comes up through
`from_components` and `bootstrap_sim_cell`, which never touch that gate, so on the simulation path run
the doctor by hand.

### 2.3 Selecting a driver, and what construction does

One key picks the driver, `robot.vendor`. Ask the config system what a key means and where its value
came from rather than trusting a table: `python -m src.config explain robot.vendor`, and
`python -m src.config where robot.ur --tier all --limit 500` for the per-vendor blocks.
`robot.workspace_limits.*`, `robot.motion_limits.*`, `robot.gripper.*` and `robot.safety.*` apply to
every vendor.

`robot.ur.model` is not cosmetic. Read its docstring with `explain`: the model keys the safety DH
chain, the exact-mesh collision bundle and the planner's robot config together. Without it a UR3e
would be planned and collision-checked against UR5e link lengths with nothing saying so.

`create_arm(vendor, config=...)` looks up the registered factory and builds the driver; the simulator
and `ur_rtde` are touched only at `connect()`. **UR `connect()` has a rollback worth knowing:** the
configured payload envelope is pushed with `setPayload`, and if that fails the driver tears the
connection back down and raises `RobotConnectionError`, on the stated grounds that an unverified
payload on a connected controller is more dangerous than no connection at all. The push is gated on
`robot.safety.payload.enforce`, so disabling the guard also stops the driver touching the controller
payload.

**Both real drivers build their own `SafetyPreflight` from config in `__init__`.** `DummyRobotArm`
carries none and simply drives: fine for tests, never for anything with a motor.

One more thing about a UR: `URRobotArm.__init__` always constructs a Robotiq `GripperController` from
`config.gripper` and exposes it as `arm.gripper`, whatever `gripper.vendor` says. It is inert, since
`connect()` does not connect it and nothing in the pick path touches the property, but do not read "I
set a different vendor" as "there is no Robotiq object here". The gripper the cell actually uses comes
from `create_gripper` on the composition path.

### 2.4 The multi-robot key registry

[`src/robot/drivers/sim/robot_models.py`](../../src/robot/drivers/sim/robot_models.py) maps one model key to a
simulator USD, a baked gripper variant, a reach and a payload for `ur3e`, `ur5e` and `ur10e`, and
`curobo_robot_yml(model)` turns the same key into `{key}.yml`. One key drives the DH table, the
exact-mesh bundle, the planner descriptor and the simulator asset selection, which is why the schema
cross-validates it (4.8).

A `ur3e` simulation cell requires `robot.sim.gripper_mount`, because the shipped `ur3e.usd` bakes no
gripper variant. Without the mount the cell comes up as a bare 6-DoF arm and the first symptom is a
late, cryptic gripper connect failure naming a driven joint that is not in the articulation's degrees
of freedom. The `ur3e` config layer sets it.

---

## 3. Declaring an end-effector

`robot.gripper.vendor` picks the driver `create_gripper` builds; `min_width_mm` and `max_width_mm`
are the physical opening floor and ceiling. `GripperVendor` has eight members: `robotiq`, `onrobot`,
`vacuum`, `jaw_io`, `dummy` and `none` have drivers, and `franka_hand` and `schunk` are reserved names
with none. Two sub-blocks configure the digital-I/O end-effectors, `robot.gripper.vacuum.*` and
`robot.gripper.jaw_io.*`, and one configures the Modbus one, `robot.gripper.onrobot.*`. Simulation
end-effector selection lives in `robot.sim.*` instead, as `gripper_variant`, `gripper_mount` and
`suction_cup`.

Three things belong here rather than in [06](06-grippers.md), because they are properties of the
composition rather than of a gripper:

**A misconfigured end-effector becomes a working `NullGripper`, not a crash.** The cell comes up,
every pick reports success, and the jaws close on nothing. Five config combinations produce it, one
per substitution reason: an unknown vendor; `robotiq` on an arm that is not a UR; `vacuum` or
`jaw_io` on an arm advertising no digital I/O; and a recognised vendor with no driver in this
repository. Each carries a `GripperSubstitution` naming the reason and the fix on the object itself.
A `NullGripper` whose `substitution` is `None` is a cell that has no end-effector on purpose.

**Connect order is arm first, then gripper.** `from_robot_config` does not call `arm.connect()`, and
both digital-I/O drivers touch the controller's I/O as soon as they themselves connect, so a real cell
must connect the arm first. Teardown is the exact reverse, because a teardown that skips the gripper
leaves a vacuum line asserted. `ConnectedCell` enforces both orders.

**The simulator's grippers are unreachable from a real cell.** No `GripperVendor` member points at
them, so `create_gripper` can never return one. The simulation runners hand-build them with a live
session, which is also why simulation uses `from_components`.

---

## 4. The `SafetyPreflight`

### 4.1 Where the preflight is built

Not in a composition root. `URRobotArm.__init__` and `KukaRobotArm.__init__` each build one;
`bootstrap_sim_cell` builds one with `sim_safety_preflight(cfg)` and injects it into `IsaacRobotArm`;
`DummyRobotArm` has none. All three real paths call the same factory,
`SafetyPreflight.from_safety_config(cfg.robot.safety, cfg.robot.workspace_limits)`, so the safety
layer is wired identically on the live simulation path and on the real-hardware path. Prove it on
your own machine:

```powershell
python -c "from src.config import load_config; from src.robot.safety import SafetyPreflight; c=load_config().robot; print(SafetyPreflight.from_safety_config(c.safety, c.workspace_limits).guard_names)"
```

On the shipped tree that prints
`('workspace', 'joint_limit', 'ik_quality', 'self_collision', 'payload', 'motion_continuity')`, and
all six `enforce` flags default true.

The simulator has one escape hatch: `bootstrap_sim_cell(..., safety: bool = True)` builds the
preflight only when `safety` is true, so `safety=False` gives an arm with no preflight at all.
Exactly one caller uses it, a wrist-camera inspection tool that does not pick, so every pick runner is
guarded. If you copy a runner's boot line, copy the default with it.

### 4.2 The ordered pipeline

Guards run in a canonical order, and `evaluate` short-circuits on the first rejection, logs it, and on
accept memoises the target for the next continuity check.

| # | Guard | `enforce` key | Needs | Refuses on |
|---|---|---|---|---|
| 1 | `workspace` | always wired | `target_pose` in `Frame.BASE` | a pose outside the box shrunk by `safety.limits.workspace_margin_mm` (default 20.0) on every face |
| 2 | `joint_limit` | `safety.joint_limits.enforce` | `target_joints`, plus `arm` for the vendor table | any axis outside `[min + margin_deg, max - margin_deg]`, `margin_deg` default 5.0 |
| 3 | `ik_quality` | `safety.ik_quality.enforce` | `target_joints`, plus `arm` | non-finite joints, DoF mismatch, joint jump, near-limit proximity, near-singularity |
| 4 | `self_collision` | `safety.self_collision.enforce` | `target_joints` and a resolvable kinematics model | any monitored pair closer than `min_distance_mm`, default 10.0 |
| 5 | `payload` | `safety.payload.enforce` | config only | negative mass, mass over `max_mass_kg`, any negative inertia component |
| 6 | `motion_continuity` | `safety.motion_continuity.enforce` | a previous **accepted** target | joint step, TCP step or orientation step over the caps |

Every rejection becomes a typed `MotionResult`, prefixed `[safety:<guard>/<reason>]`, with one
`MotionStatus` per `SafetyReason` from a closed table in
[`src/robot/safety/decision.py`](../../src/robot/safety/decision.py). A guard that cannot decide reports
`unavailable`, mapped to `CONTROLLER_REJECTED`.

### 4.3 The fail-closed mechanism

`SafetyDecision.unavailable()` constructs with `accepted=False`. A guard that cannot decide therefore
rejects. There is no "could not check, waving it through" path in this pipeline. The authority
argument, that safety outranks the deterministic geometry, the recovery layer and the optional learned
layers, and that those may only reorder or filter already-cleared candidates, is in
[safety-math.md](../safety-math.md).

### 4.4 What each guard does not cover

The formulas are in [safety-math.md](../safety-math.md). The holes are here, and they are the half
that gets forgotten.

**Workspace** is an axis-aligned box test on the endpoint position, nothing else: not orientation,
and not the *path* between two accepted poses. A joint-only command with no `target_pose` is accepted.
A margin that inverts an axis raises at construction, so the failure lands at boot rather than at move
time.

`robot.safety.limits.enforce` is a **dead flag**. It is in the schema, defaulted true and set in
`robot.yaml`, but only `workspace_margin_mm` is read from that block: the workspace guard is
unconditionally wired and only the other five families consult an `enforce`. Setting it false does not
disable the workspace guard. To widen the box edit `robot.workspace_limits`; to remove the margin set
`workspace_margin_mm: 0.0`.

**Joint limit** resolves its envelope in priority order: static `min_deg` and `max_deg` from YAML,
then the built-in UR table keyed on `capabilities.vendor == "ur"` *and* `.model`, then nothing, which
means unavailable and a fail-closed rejection. That table is the manufacturer factory envelope, not
your installation limits, because no driver here reads joint limits from controller telemetry. So the
guard will pass a configuration the controller protective-stops on. The controller is the tighter
backstop.

Any cell whose vendor is not `ur` must supply `min_deg` and `max_deg` in YAML, or every motion is
refused. That includes the simulator, whose capability vendor is `sim`, and every KUKA cell. The `sim`
layer supplies the UR factory envelope. Do not paste those numbers onto a KRC cell: they are far wider
than any KR axis and would re-create the hole this guard closes. KUKA per-axis limits come from the
controller or the datasheet, and `config/robot/templates/kuka_eki.yaml` ships no `joint_limits` block.

**IK quality** runs five checks: non-finite joints, DoF mismatch, joint jump from `current_joints`,
limit proximity and near-singularity. The singularity test needs an arm advertising `has_native_fk`,
and a failed Jacobian probe returns unavailable rather than a silent pass. Two holes: no
`current_joints` means no jump check, and limit proximity is skipped when no envelope resolves, so a
KUKA cell with no static table silently gets no proximity coverage.

**Payload** validates the *declared* envelope and cannot know what is bolted to the flange. Its three
checks duplicate the schema validators on purpose: a config assembled by a path that bypassed schema
validation, such as a grasping preset overlay or a hand-built test config, would otherwise reach the
controller with a bad payload.

**Motion continuity** bounds three step sizes against the memoised previous accepted target. The first
command after `reset()` always passes, and a DoF mismatch skips the joint check rather than rejecting.
It bounds the net delta between two commanded endpoints, and the interpolator's path between them is
not examined by this guard at all. That is what the continuous monitor in 5.4 is for.

### 4.5 The two documented skip-sets

Both are code constants, deliberately not operator YAML, because a typo in a skip-set silences a
guard.

`gate_joint_target` skips `workspace`, `ik_quality` and `motion_continuity`: a commanded joint move to
a park, home or scan pose is a deliberate trajectory restart, so only the destination-safety guards
run and the continuity memo is reset on both sides.

The planner path skips `ik_quality` and `motion_continuity` only. Those two proxy "the blind
interpolator will not teleport", which is meaningful only when a move lerps in joint space. A planner
owns a continuous, dynamically feasible, collision-checked path, so a large net joint delta such as a
shoulder wrap from `+pi` to `-pi`, physically about zero rotation, is not a teleport. Workspace, joint
limits, self-collision and payload all still run on the planned final configuration, so the exact-mesh
gate is never skipped there.

The difference between the two matters on a real cell. A real UR gates the planner's final
configuration through `gate_joint_target`, whose skip-set **includes `workspace`**. So on a real UR the
planned target is checked for joint limits, self-collision and payload and **not** against the
workspace box. If that box is what keeps the arm out of a fixture, it is not what stops a planned move;
declare the fixture instead.

### 4.6 `enforce: false` does more than expected

`enforce=False` removes the guard from the pipeline at construction. It does not merely silence the
log; the safety surface for that family is gone until the flag comes back. `SafetyPreflight` names
what was left out through `omitted_guards`, and `guard_names` says what is present, so print one of
them (4.1) rather than assuming.

### 4.7 The two silent-hole keys

`self_collision.kinematics_model` looks optional and is load-bearing. Unset on an arm whose vendor is
not `ur`, such as the simulator, which is physically a UR5e, the arm capsules resolve to `None` and the
guard checks only base, tool and fixtures, with no arm against arm at all, and with no error. The same
gate exists on the exact-mesh path.

`self_collision.kinematics_base_yaw_deg` looks cosmetic at its 0.0 default. The bundled UR DH base is
rotated 180 degrees about Z from the simulator's UR5e `base_link`, and because the reconcile is a pure
base rotation every inter-link distance is preserved: arm against arm still looks fine while every
**fixture** check is silently nonsense ([safety-math.md](../safety-math.md)). The `sim` layer sets
both; the base tree leaves them unset. Check with
`python -m src.config explain robot.safety.self_collision.kinematics_model --profile sim`.

### 4.8 Cross-validators that fire at config load

[`robot_schema.py`](../../src/config/schema/robot/robot_schema.py) refuses three shapes of
half-migrated cell. On an enabled simulation cell, `self_collision.kinematics_model` must equal
`sim.robot_model`; on `vendor: ur` it must equal `ur.model`; and setting it on a vendor that is
neither `ur` nor `sim` is an error. Profiles compose, so a chain that mixes a simulator layer onto a
KUKA cell would otherwise load cleanly with `kinematics_model: ur5e`, after which the guard evaluated a
KUKA arm against UR5e link lengths.

---

## 5. Self-collision: two backends, and what each is worth

### 5.1 The two backends

The config surface is one block: `python -m src.config where self_collision --tier all --limit 500`.

**`capsule`** is the fallback and needs nothing installed: a chain of capsules (default link radius
60 mm) plus a base column and a tool capsule rooted at the commanded TCP, closed form and
sub-millisecond ([safety-math.md](../safety-math.md)). `tool_model: capsule`, the default, is a
rotation-invariant bounding cylinder along the approach axis with radius `tool_radius_mm`, default
70.0, and it is over-conservative against a bin wall: a two-finger gripper reaching into a tote is
about 27 mm wide perpendicular to its closing axis while a 70 mm radius claims 140 mm in every
direction, so an off-centre target is falsely wall-rejected. `tool_model: finger` is a thin capsule
along the grasp's closing axis and is rotation-aware. The structural false negative: the capsule
backend skips any link pair within two indices, the wrist cluster, because capsules with realistic
radii always overlap there.

**`fcl`** is exact mesh distance, against committed per-link meshes that are DH-baked and validated
against the simulator's own USD. Pairs within one DH frame are skipped and every farther pair is
checked exactly, so the wrist pairs the capsule backend has to skip are covered, and each link is also
checked against the declared fixtures. Coal is preferred and `python-fcl` is the guarded fallback; the
swap is behaviour-identical, because both run the same meshes, pairs, thresholds and distance query.

**`fcl` is the shipped default**, and the reason is a structural hole in the proxy rather than accuracy
in the abstract: the proxy roots the tool capsule in `target_pose`, and `gate_joint_target` never sets
one. So on every commanded joint move, and on a planner's final-configuration gate, the proxy does not
model the gripper at all. The two backends disagree on a substantial fraction of random
configurations, and the disagreement runs one way: the proxy accepts configurations whose true mesh
distance is zero. The price is roughly two orders of magnitude in per-check time, which is still small
against a move.

### 5.2 Does this robot actually have exact-mesh authority?

Bundles are per robot, `{model}_collision_meshes.npz`, and a present ur5e bundle says nothing about a
UR3e cell. Three ship in [`safety/data/`](../../src/robot/safety/data/): `ur5e`, `ur3e`, and a
`schunk_egu50` variant, which is the ur5e arm with a different gripper. A new model gets exact meshes
as soon as its bundle lands beside them, with no code change.

Call `mesh_backend_status(model)` from `src.robot.safety._fcl_self_collision` for each model that
matters. The tokens are `ok`, `unknown_model`, `no_bundle`, `variant_model_mismatch` and `no_engine`.
`unknown_model` means there is no bundled DH row for that key, so anything that is not a UR has no
exact-mesh authority, ever. `no_bundle` is recoverable by baking one with
[`scripts/isaac/bake_ur_collision_meshes.py`](../../scripts/isaac/bake_ur_collision_meshes.py) under
the simulator's interpreter; it is self-validating, and that is the gate, so run it for a model whose
bundle already ships, without writing, and only trust a new model if that reproduces the committed
bundle.

### 5.3 Asking for `fcl` does not guarantee getting it

**This guard falls back rather than failing closed.** If the engine or the per-model bundle is
missing, it logs one warning naming the reason and then runs the capsule proxy. It never silently
disables the guard; it silently weakens it, loudly. Two consequences worth carrying: a cell that is not
a UR has no bundle and therefore always runs the proxy, and no bundle carries base geometry, so while
the mesh path runs, the declared pedestal column (`base_radius_mm`, `base_height_mm`) is not modelled.

The schema docstring for `SelfCollisionSafetyConfig` still describes a version of this that reports
unavailable and refuses while `enforce` is true. The code does not do that. The code is authoritative.

Check what actually resolved with `python -m src.robot.safety.planning --check`.

The practical consequence: **do not accept a capsule verdict as evidence about a real cell.** Turn
`backend: fcl` on, set `kinematics_model` and `kinematics_base_yaw_deg` for your arm (4.7), consider
`tool_model: finger`, add the bin walls as `fixtures`, and set `planner_margin_mm` per section 6. If
`mesh_backend_status(model)` is not `ok`, treat that cell as having no self-collision authority.

### 5.4 The two opt-in path gates

`safety.trajectory_check` gates every configuration of a planned path before any of it is commanded,
with the same guards a joint move gets: joint limits, self-collision including fixtures, and payload.
It is off by default. While it is off, nothing examines the middle of a plan, so a path that grazes a
fixture halfway and lands clear passes every check there is. `stride` samples the path instead of
checking all of it, and the final configuration is checked whatever the stride.

[`src/robot/safety/continuous_monitor.py`](../../src/robot/safety/continuous_monitor.py) runs the exact-mesh
backend over **every** interpolation waypoint of a move, with a clearance margin and a fail-safe of its
own: a check that overruns its budget or cannot run returns a stop or a hold, never a continuation
without a result. `ContinuousGuardProfile` defaults to `enabled=False`, `margin_mm=8.0` and
`max_check_ms=12.0`, and it is in no pipeline by default; the simulation runners wire it through
`wire_safety_guards(arm, continuous_guard=True, ...)`. The margin has a real ceiling: a natural grasp
puts the arm's own `wrist_1` and `wrist_3` pair at about 19.6 mm, so a margin above that false-stops a
good pick.

---

## 6. The motion-planning requirement

Read this before reporting any pick rate. The safety argument is that motion is collision-**planned**
by cuRobo and collision-**checked** against exact meshes by Coal or `python-fcl`. A missing engine does
not degrade that argument gracefully. It replaces it: blind IK proposes self-colliding branches the
guard then correctly rejects, which reads as bad *grasping*, and the capsule proxy accepts what the
mesh check refuses.

Neither engine can be a line in `requirements.txt`. cuRobo needs its own Python 3.10 interpreter,
because one process holds exactly one `warp` and the simulator ships a version cuRobo cannot use, and
Coal has no Windows wheel, which is why `python-fcl` is the guarded fallback and does ship. Install
both from the page that owns them and is named in the failure banner,
[`ext_deps/README.md`](../../ext_deps/README.md); `scripts/ext_deps/install.ps1` puts them where the
defaults look. Then probe, before anything else:

```powershell
python -m src.robot.safety.planning --check     # paths only, milliseconds
python -m src.robot.safety.planning --doctor    # actually loads both engines, seconds
```

Run `--check` before spending a minute on a simulator boot, but do not trust it alone: it reads paths,
and it can report available while an operating-system policy refuses one of the sidecar's libraries,
with a collapsed pick rate as the only symptom. `--doctor` catches that and exits 2 when an
application-control policy is the cause ([docs/code-integrity.md](../code-integrity.md)). Exit 1 is not
a broken install, it means not fully anchored, which is the expected reading on a machine with no GPU
environment; exit 0 does not prove cuRobo runs, because the check only verifies the environment's
interpreter exists. The command reads the model from config and also takes `--model`, so a UR3e cell
needs no guess: `python -m src.robot.safety.planning --check --model ur3e`.

**The two drivers take opposite policies on a missing engine, on purpose.** A real UR fails closed:
`robot.ur.motion_planner` is a real config key (`ik` or `curobo`, shipped `curobo`), and with cuRobo
unavailable the move returns `CONTROLLER_REJECTED`, while no collision-free plan returns `TIMEOUT`. It
never falls back to blind IK. The simulator degrades: its planner is a dataclass field rather than a
YAML key, which `python -m src.config explain robot.sim.motion_planner` will tell you outright, and
with cuRobo missing the resolution latches unavailability, emits one warning and returns `ik`. Choose
the simulation planner from the runner instead, through `--motion-planner`.

That silence is what the boot gate stops. `require_motion_stack` in
[`src/willy_sim/harness/bootstrap.py`](../../src/willy_sim/harness/bootstrap.py) runs before the simulator
boot, keyed on the cell's actual descriptor and `kinematics_model`, and raises
`DegradedMotionStackError` unless `WILLY_ALLOW_DEGRADED_MOTION=1`. It prints a box rather than a log
line, because a simulator boot emits thousands of lines and a one-line warning is invisible in
practice, and it states the consequence: any pick rate measured now is not the pick rate of the
configured system and must not be reported as one. It is scoped honestly, since a cell whose
`self_collision.backend` is not `fcl` is not held to the mesh engine and one asking for `ik` is not
held to cuRobo. Reproduce it off-box:

```powershell
python -c "from src.willy_sim.harness.bootstrap import require_motion_stack; require_motion_stack('curobo', robot_config='ur5e.yml', kinematics_model='ur5e', exact_mesh_collision=True)"
```

**The planner-margin handshake.** cuRobo plans against spheres; the guard re-checks exact meshes.
Nothing tells the planner about `min_distance_mm`, so without a margin it keeps returning
configurations sitting just inside the guard's threshold, which the guard then refuses, which reads as
bad grasping rather than as a rejected plan. `apply_self_collision_margin` in
[`src/robot/safety/planning/_curobo_margin.py`](../../src/robot/safety/planning/_curobo_margin.py) adds half the
margin to each link's self-collision buffer, because the planner subtracts both links' buffers from a
pair distance.

`planner_margin_mm` is per robot and must be measured, not copied. The shipped layers record 10.0 for
the UR5e in `robot.sim.yaml` and 4.0 for the UR3e in `robot.ur3e.yaml`, each with its reasoning in the
comment: a thinner-linked arm reads as permanently self-colliding once every sphere is inflated to the
larger arm's value, and finds no plan at all. `robot.ur5e.yaml` leaves the key at its 0.0 default and
says why. Read those comments before setting yours. The honest limit is stated in the module: spheres
are not meshes, so the buffer is a cushion, not a proof, and it is the guard, never the planner, that
decides what actually executes. The wire contract and the real-UR waypoint execution are in
[`src/robot/safety/planning/README.md`](../../src/robot/safety/planning/README.md). That round trip is
unexercisable off a real cell; the logic is exercised with a fake planner.

---

## 7. Bring-up, and where to look when it goes wrong

`python -m src.robot.execution.real_cell --check` is the checklist, and it touches nothing. Run
against the shipped tree as a UR cell it reports **three blocking items**, and that is the correct
default state of a freshly configured real cell rather than a defect:

| Blocking item | What it looks like at the bench if you skip it |
|---|---|
| `gripper.tool_frame.source` is `undeclared` | nobody has said where the grasp centre sits on the flange, so a top-down grasp commanding z = 37 mm drives the flange there and the fingertips through the bench |
| `safety.payload` has `enforce: true` and `mass_kg: 0.0` | `connect()` refuses this outright, which is better seen at a desk than after driving to the cell. It would otherwise push a zero payload and leave the controller's protective-stop model under-reading a mounted tool |
| no `CAMERA->BASE` resolver | perception reports grasps in the camera frame; without a resolver the driver rejects every motion as `INVALID_TARGET`. Fail-closed and correct, and at the bench it looks exactly like a cell that hangs |

[`config/robot/robot.ur5e.yaml`](../../config/robot/robot.ur5e.yaml) leaves exactly those three unset
**on purpose**, as the worked example for a real bench, and carries each one as a commented block
saying what to measure. Its own rule decides which keys carry a value: a wrong value that fails closed
ships as an example with its assumption stated, so a workspace box that is too small refuses a motion
visibly and the operator widens it; a wrong value that fails open does not ship at all, because a
plausible tool frame or payload drives the arm into the bench and logs a success.

A blocking item does not necessarily stop the connect. A cell with a blocking checklist can connect
and then refuse every motion, which is the failure that reads as a broken robot. What refuses the
connect is the driver's own preflight, not this checklist. Four further items are warnings and never
block, and two are deferred to the bench because no interface can answer them.

The order that works:

1. `python -m src.robot.drivers.doctor --require ur`, the SDK is installed. Exit 0 is the pass.
2. `python -m src.robot.execution.real_cell --check`, fix everything blocking.
3. `python -m src.robot.safety.planning --doctor`, the planner environment, before any motion.
4. `python -m src.robot.perception --prompt "..."`, the camera and the models, with no robot.
5. Calibrate each camera, one command per rig ([03](03-calibration.md)), then `--dry-run`, then
   `--runs 1`, then a campaign.

Along the way: confirm the mesh bundle for this robot (5.2); turn the exact-mesh guard on and tell it
which kinematics it guards (5.3); choose the planner and measure `planner_margin_mm` on this robot
(6); verify the guard pipeline with the `guard_names` one-liner (4.1) and expect all six, since
anything missing means an `enforce: false` removed that safety surface; and for any vendor that is not
`ur`, supply static joint limits or every move is refused (4.4).

Re-measure after any of the above, because margins, the yaw reconcile and the planner choice all change
which motions are proposed and which are accepted.

Logs land in `logs/robot/`, with a per-module file under `logs/robot/modules/`, but the self-collision guard and the simulator driver's
planner resolution log on the plain module logger, so read the runner's standard output too. Four
symptoms cover most of the confusion here:

- every move returning `CONTROLLER_REJECTED` on a KUKA, simulator or dummy-capability cell is the
  joint-limit guard failing closed with no table (4.4);
- picks that look like bad grasping are usually the blind-IK fallback proposing self-colliding
  branches the guard then correctly rejects (6);
- collisions slipping through a cell configured `backend: fcl` mean the engine or bundle is missing
  and the capsule proxy is running with its wrist skip (5.3);
- a `vacuum` or `jaw_io` gripper that became a `NullGripper`, or whose `connect()` raised with no I/O,
  is either an arm that does not advertise `SupportsDigitalIO` or a gripper connected before the arm
  (section 3).

---

## 8. What this layer does not give you

Stated once in the code and repeated in
[`src/robot/safety/planning/README.md`](../../src/robot/safety/planning/README.md) and in
[safety-math.md](../safety-math.md):

> This is a software collision-**avoidance** layer, not "the safety system". It **reduces** collision
> risk in simulation. The real-hardware safety **guarantee** is independent, certified functional
> safety: hardware emergency-stop circuits plus ISO 10218, ISO/TS 15066 and ISO 13849, which run
> independently of this software. A green simulation result is not a commercial safety claim.

Never validated on hardware: a UR executing this driver on a physical arm, as opposed to controller
software in a container; KUKA, which additionally has no built-in joint-limit table and no bundled DH
chain, so no exact-mesh authority and no arm-against-arm capsules; the real planner round trip; the
wiring of both digital-I/O grippers; and any safety guarantee at all.

Validated in simulation: the six-guard preflight on every pick, the planner as the simulation default
gated by the mesh guard, the exact-mesh backend in the `sim` layer, the planner-margin handshake, the
fail-closed motion-stack gate and its banner, and the simulator jaw and suction grippers with their
measured profiles.

Next: [05](05-pick-loop.md), what happens above this layer once a motion is allowed, and
[06](06-grippers.md), making an end-effector move.
