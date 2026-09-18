# 4. Robot, drivers and the safety stack

You have a config tree that validates ([01](01-configuration.md)). This guide covers what stands between
that YAML and a moving arm: which driver is built, which hand, which guards run in which order, and which
two external engines the motion stack needs before its answers mean anything.

```python
from willy import Robot, SafetyPreflight, load_tree

tree = load_tree()                   # the cell WILLY_PROFILE names
robot = Robot.from_tree(tree)        # the arm and the hand, built and not connected
print(robot)                         # arm, hand, lock, safety, planner route, camera world
print(robot.safety())                # what the built arm will refuse
print(SafetyPreflight.from_tree(tree).guard_names)
```

At a desk, `load_tree("console_dummy")` builds a dummy arm and a dummy hand. Its safety line reads
`UNGATED`, because a dummy arm carries no preflight, while the preflight built from the same tree still
lists its guards. The desk checklist for your own cell is
[`examples/real_robot/02_check_the_cell_at_a_desk.py`](../../examples/real_robot/02_check_the_cell_at_a_desk.py).

The mathematics lives in [docs/safety-math.md](../safety-math.md): capsule geometry, the DH chain, the
base-yaw reconcile, the Jacobian singularity test, exact mesh distance and the fail-closed authority rule.
Driving a hand is [06](06-grippers.md).

**Two facts to start from.** The default grasp attempt is open-loop: the decision gate, closed-loop
refinement and verification, recovery, fusion with its commit gate, the learned ranker and the learned
success model all ship `enabled: false` ([05](05-pick-loop.md)). The safety layer is the exception: the
driver builds its `SafetyPreflight`, or has one injected, so it runs on every live path (section 4.1).

**Prerequisites.** A virtual environment with `requirements.txt`, in which `python -m src.config` exits 0.
`python` means that interpreter throughout. Only a simulator run needs the simulator's own interpreter.

---

## 1. The vendor contract

### 1.1 `RobotArm`

[`src/robot/core/robot_arm.py`](../../src/robot/core/robot_arm.py) defines a `@runtime_checkable` Protocol.
Every driver implements it, and nothing above the driver layer imports a driver class.

| Group | Members | Contract |
|---|---|---|
| introspection | `capabilities`, `is_connected` | vendor, model, DoF, feature flags |
| lifecycle | `connect`, `disconnect` | idempotent both ways |
| state | `get_tcp_pose`, `get_joint_positions` | pose tagged `Frame.BASE`; joints in radians |
| motion | `move_joint`, `move_linear`, `stop` | raise `RobotMotionRejected` or `RobotConnectionError` |
| motion | `move_to_joints`, `move` | return a typed `MotionResult` |
| kinematics | `fk`, `ik` | BASE frame both ways |
| bool shims | `is_inside_workspace`, `move_to`, `move_home`, `wait_until_steady` | return `False` instead of raising |

The units are fixed: millimetres, XYZW quaternions, frame-tagged poses, joints in radians. Vendor
conventions, such as UR's axis-angle rotation vectors and KUKA's E6POS with ABC Euler degrees, stay inside
the driver package.

Prefer `move()` over `move_to()`. The bool pair folds every failure into one `False`, so a caller cannot
tell a workspace rejection from an IK failure from a controller refusal. `move()` returns a `MotionResult`
whose `MotionStatus` comes from a closed set, with one safety rejection per guard:

```powershell
python -c "from src.robot.core import MotionStatus; print([m.value for m in MotionStatus])"
```

`Robot.move`, `Robot.move_joints` and `Robot.home` call these typed verbs and return a `MotionReport`
([06](06-grippers.md), section 6).

### 1.2 The three optional capability Protocols

Some features only some controllers have. They live as separate `runtime_checkable` Protocols in
[`src/robot/core/arm_capabilities.py`](../../src/robot/core/arm_capabilities.py), and a caller checks for
one with `isinstance()`.

| Protocol | What it adds |
|---|---|
| `SupportsDigitalIO` | `set_digital_output`, `get_digital_input`, analog out, banked by `DigitalIOPort` (`standard`, `configurable`, `tool`) |
| `SupportsForceTorque` | `get_tcp_wrench` in N and Nm, BASE frame, plus `get_joint_torques` |
| `SupportsRobotStatus` | `get_robot_status` and `recover_from_protective_stop` |

The vendor-neutral `RobotMode` and `SafetyMode` enumerations turn a controller's integer status codes into
portable words; the UR driver maps its integers in `src/robot/drivers/ur/arm.py`. `URRobotArm` is the only
driver that implements any of the three. `SupportsDigitalIO` carries weight: the two digital-I/O
end-effectors run on it, and neither is built on an arm that does not advertise it.

### 1.3 `Gripper` and `ObjectDetectingGripper`

[`src/robot/core/gripper.py`](../../src/robot/core/gripper.py) defines `is_connected`, `min_width_mm`,
`max_width_mm`, `connect`, `disconnect`, `activate`, `set_width_mm(width_mm, *, speed=None, force=None)`
and `get_width_mm()`. Widths are millimetres. `force` and `speed` are normalised to `[0.0, 1.0]`, not
newtons and not mm/s, and each driver maps them to its own units. The OnRobot driver is the exception: it
takes force in newtons through its own config key.

`ObjectDetectingGripper` adds one method, `is_object_detected()`. A driver that implements it opts the cell
into post-close verification; a driver that does not is trusted after the close command. That is the
documented default. Which drivers answer, and on what evidence, is in
[`src/robot/grippers/README.md`](../../src/robot/grippers/README.md).

### 1.4 How vendor neutrality is enforced

No vendor SDK is imported at module import time. The registry stores factory callables with the driver
import inside the factory body, so importing `src.robot.drivers` works on a machine with no `ur_rtde`.
`create_arm` checks its result against the `RobotArm` Protocol, so a faulty factory cannot return the
wrong shape. The UR driver guards `rtde_control` and `rtde_receive` in
`src/robot/drivers/ur/connection.py` and fails only at `connect()`. It guards `rtde_io` and the dashboard
client separately, so I/O and safety status can degrade without taking the move core with them.

---

## 2. The drivers

### 2.1 What is registered, and what it is worth

| Vendor | Registered | Needs SDK | Evidence |
|---|---|---|---|
| `ur` | yes | `rtde_control`, `rtde_receive` | measured against real controller software (URSim): connect, power, motion, digital I/O, tool frame, payload, protective stop |
| `kuka` | yes | none (EKI and KRL over TCP and XML) | never touched hardware; an integrator deploys the KRL program in [`config/robot/templates/kuka/`](../../config/robot/templates/kuka/) first |
| `sim` | yes | `isaacsim` | measured in simulation: the only path that has run full motion |
| `dummy` | yes | none | a desk arm with **no kinematics**: `fk` returns the last recorded pose, `ik` the last recorded joints |
| `franka`, `ros2` | **no** | | empty slots; `create_arm` raises `RobotConnectionError` |

Bring-up per vendor: [`src/robot/drivers/README.md`](../../src/robot/drivers/README.md) and the vendor
READMEs beside it.

### 2.2 Probe the host with the driver doctor

```powershell
python -m src.robot.drivers.doctor               # one row per arm and gripper vendor
python -m src.robot.drivers.doctor --require ur  # exit 1 unless the ur driver is ready here
```

Each row says whether the vendor is registered, its SDK imports and the driver is ready. The doctor
imports each SDK rather than asking `find_spec`, because a package whose native extension the operating
system refuses to load would otherwise read as ready. `isaacsim` is the one exception: importing it takes
tens of seconds and starts a renderer, so it is resolved by spec. A row that is not ready names the reason
per module, so "installed and will not import" reads differently from "not installed". The plain table
exits 0, `--require <vendor>` exits 1 when that vendor is not ready and 2 on an unknown vendor, and
`--json` prints the same records as JSON. Read your own machine rather than a pasted transcript.

The `registered` column comes from the registries themselves. To check by hand, `src.robot.drivers`
exports `available_vendors` for arms, and `available_gripper_vendors` lives in `src.robot.grippers`.

The same readiness gate runs whenever an arm is built from config. `resolve_arm` in
[`src/robot/execution/robot_parts.py`](../../src/robot/execution/robot_parts.py) calls
`require_arm_vendor_ready`, and both the pick service and `Robot.from_tree` build their arm through it, so a
misconfigured host fails there rather than inside `connect()`. The dummy vendor and a simulator in mock
mode are exempt, so the offline tests are never gated. A simulator runner that builds through
`from_components` does not reach the gate; on that path, run the doctor by hand.

An arm the caller hands in is exempt only when it advertises another vendor than the config names, because
such a stand-in drives no device of that vendor. A handle that advertises the configured vendor still faces
the gate: a `URRobotArm` constructs on a machine with no `ur_rtde`, and the missing SDK surfaces only inside
`connect()`.

### 2.3 Selecting a driver, and what construction does

One key picks the driver: `robot.vendor`. Ask the config what a key means and where its value came from:

```powershell
python -m src.config explain robot.vendor
python -m src.config where robot.ur --tier all --limit 500
```

`robot.workspace_limits.*`, `robot.motion_limits.*`, `robot.gripper.*` and `robot.safety.*` apply to every
vendor.

`robot.ur.model` is not cosmetic. It selects the safety DH chain, the exact-mesh collision bundle and the
planner's robot descriptor together. With the wrong model, a UR3e is planned and collision-checked against
UR5e link lengths and nothing says so (`python -m src.config explain robot.ur.model`).

`create_arm(vendor, config=...)` builds the driver from its registered factory; the simulator and `ur_rtde`
are touched only at `connect()`. A UR `connect()` pushes the configured payload with `setPayload`. If the
push fails, the driver closes the connection again and raises `RobotConnectionError`, because an unverified
payload on a connected controller is worse than no connection. The push is gated on
`robot.safety.payload.enforce`, so disabling the guard also stops the driver touching the controller's
payload.

**Both real drivers build their own `SafetyPreflight` from config.** `DummyRobotArm` carries none and simply
drives: fine for tests, never for anything with a motor.

`URRobotArm` is the arm alone and has no `gripper` property. The hand a UR cell drives is the one
`build_gripper` in [`src/robot/execution/robot_parts.py`](../../src/robot/execution/robot_parts.py) builds
from `gripper.vendor`. A cell configured for suction or for I/O jaws holds no Robotiq object, and a Robotiq
cell holds exactly one `GripperController`.

### 2.4 The multi-robot key registry

[`src/robot/drivers/sim/robot_models.py`](../../src/robot/drivers/sim/robot_models.py) holds every UR from
the UR3 to the UR10, both series. One model key maps to a simulator USD, a reach and a payload, and
`curobo_arm_descriptor(model)` turns the same key into `willy_{key}.yml`, the planner's descriptor of the
arm. The descriptor carries no hand: the planner adds the hand `robot.gripper.model` names as a body link
when its sidecar starts, placed by the declared tool frame. One key drives the DH table, the exact-mesh
bundle, the planner descriptor and the simulator asset, which is why the schema cross-validates it (4.8).

In simulation the mount follows from `robot.gripper.model`. The `ur5e` and `ur10e` assets bake the 2F-85
as a variant; on the other arms the simulator mounts the named hand standalone. A hand the simulator has no
mount for is refused by name before the simulator boots. A customer's own hand runs its chain on a real
cell, which never asks for a mount. A mount needs the hand's USD asset, the rotation that puts its approach
on wrist +Y and a measured `tcp_offset_mm` ([`src/willy_sim/grippers.py`](../../src/willy_sim/grippers.py)).

---

## 3. Declaring an end-effector

`robot.gripper.vendor` picks the driver `create_gripper` builds. `GripperVendor` has drivers for `robotiq`,
`onrobot`, `vacuum`, `jaw_io`, `dummy` and `none`; `franka_hand` and `schunk` are reserved names with no
driver. `robot.gripper.vacuum.*` and `robot.gripper.jaw_io.*` configure the digital-I/O end-effectors, and
`robot.gripper.onrobot.*` the Modbus one. In simulation, the hand on the arm follows from
`robot.gripper.model` and `robot.sim.robot_model`, and `robot.sim.suction_cup` picks a suction cup.

`robot.gripper.model` names the hand by its registry file, `config/grippers/<model>.yaml`, which
`src.config.grippers.load_gripper` reads. The repository ships `robotiq_2f85`, `robotiq_hande` and
`schunk_egu50`. A named hand fills `robot.gripper.max_width_mm`, `min_width_mm`, `closed_width_mm` and
every `robot.grasping.gripper_geometry` number a profile leaves unset. A profile that states one differently
is refused at load, naming the key, both values and both files. `min_width_mm` is a policy floor, admitted
from the hand's own floor up to its aperture, and `finger_pad_overlap_mm` is admitted at or above the
hand's. The self-collision guard and the planner load the hand's bundle and sphere map by that name, and the
deep calculator and the simulator mount take the hand from it too. The schema cannot see the registry, but
loading the tree can: an unknown name, a short name such as `2f85` and a registry file that cannot be read
are all refused, and `python -m src.config` says so. A hand this repository does not ship gets its body,
sphere map, retract and evidence through [your_own_gripper.md](../runbooks/your_own_gripper.md).

Three facts belong to the composition rather than to one gripper, so they sit here and not in
[06](06-grippers.md).

**A hand that cannot be built is refused at the connect.** The build puts a `NullGripper` carrying a
`GripperSubstitution` on the flange, with one reason per case: an unknown vendor; `robotiq` on an arm that
is not a UR, or on a UR with no controller address; `vacuum` or `jaw_io` on an arm with no digital I/O; a
recognised vendor with no driver. The connect reads that record before it commands the arm and raises
`NoRealGripper` with the reason and the fix. `gripper.vendor: none` builds a `NullGripper` with no
substitution: a cell with no end-effector on purpose, and it connects.

**Connect order is arm first, then gripper.** Both digital-I/O drivers touch the controller's I/O as soon
as they connect, so the arm has to be up first. Teardown runs in reverse, because a teardown that skips the
gripper can leave a vacuum line asserted. `Robot.connected()` and `Cell.connected()` enforce both orders.

**The simulator's grippers cannot be reached from a real cell.** No `GripperVendor` member points at them,
so `create_gripper` never returns one. The simulation runners build them with a live simulator session,
which is also why most runners build the service through `from_components`.

---

## 4. The `SafetyPreflight`

### 4.1 Where the preflight is built

In the driver, not in a composition root. `URRobotArm` and `KukaRobotArm` build one in their constructor;
`bootstrap_sim_cell` builds one with `sim_safety_preflight(cfg)` and injects it into `IsaacRobotArm`;
`DummyRobotArm` has none. Each of them calls `SafetyPreflight.from_safety_config`, so the safety layer is
the same on the simulation path and on the real-hardware path. `SafetyPreflight.from_tree` builds the
preflight the way the cell's driver does:

```python
from willy import SafetyPreflight, load_tree

gate = SafetyPreflight.from_tree(load_tree("console_dummy"))
print(gate.guard_names)       # what runs, in order
print(gate.omitted_guards)    # the families an enforce: false left out
```

That prints `('workspace', 'joint_limit', 'ik_quality', 'self_collision', 'payload', 'motion_continuity')`
and `()`, and every `enforce` flag defaults to true. On a tree whose exact-mesh guard needs a hand and names
none, such as the base tree, `from_tree` refuses with a `ConfigError` naming `robot.gripper.model`.

The simulator has one way out: `bootstrap_sim_cell(..., safety=False)` builds an arm with no preflight at
all. Only a wrist-camera inspection tool that does not pick uses it. If you copy a runner's boot line, keep
the default.

### 4.2 The ordered pipeline

The guards run in a fixed order. `evaluate` stops at the first rejection and logs it; on accept it
remembers the target for the next continuity check.

| # | Guard | `enforce` key | Needs | Refuses on |
|---|---|---|---|---|
| 1 | `workspace` | always wired | `target_pose` in `Frame.BASE` | a pose outside the box shrunk by `safety.limits.workspace_margin_mm` (default 20.0) |
| 2 | `joint_limit` | `safety.joint_limits.enforce` | `target_joints`, plus `arm` for the vendor table | an axis outside `[min + margin_deg, max - margin_deg]`, `margin_deg` default 5.0 |
| 3 | `ik_quality` | `safety.ik_quality.enforce` | `target_joints`, plus `arm` | non-finite joints, DoF mismatch, joint jump, near-limit proximity, near-singularity |
| 4 | `self_collision` | `safety.self_collision.enforce` | `target_joints` and a kinematics model | any monitored pair closer than `min_distance_mm`, default 10.0 |
| 5 | `payload` | `safety.payload.enforce` | config only | negative mass, mass over `max_mass_kg`, a negative inertia component |
| 6 | `motion_continuity` | `safety.motion_continuity.enforce` | a previous **accepted** target | a joint, TCP or orientation step over its cap |

The planner's joint envelope is narrower than guard 2's on purpose. Guard 2 enforces the Universal Robots
factory limit of 360 degrees per axis either way, less `margin_deg`. A planner descriptor holds the elbow at
the published planning limit of 180 degrees and every other joint at 360, and the planner narrows all of
them by `position_limit_clip`, 0.1 rad or 5.73 degrees, when it loads. Since 5.73 is more than 5.0, the
planner never proposes a configuration the guard refuses. `tests/test_planner_joint_envelope.py` asserts
that for every bundled arm and for the simulation cell, and a cell that raises `margin_deg` above 5.73
fails it.

Every rejection becomes a typed `MotionResult` prefixed `[safety:<guard>/<reason>]`, with one
`MotionStatus` per `SafetyReason` from a closed table in
[`src/robot/safety/decision.py`](../../src/robot/safety/decision.py). A guard that cannot decide reports
`unavailable`, mapped to `CONTROLLER_REJECTED`.

### 4.3 The fail-closed mechanism

`SafetyDecision.unavailable()` constructs with `accepted=False`, so a guard that cannot decide rejects. No
path in this pipeline lets a motion through because a check could not run. Safety outranks the
deterministic geometry, the recovery layer and the optional learned layers, which may only reorder or
filter candidates the guards have cleared. The argument is in [safety-math.md](../safety-math.md).

### 4.4 What each guard does not cover

The formulas are in [safety-math.md](../safety-math.md). The gaps are here.

**Workspace** tests the endpoint position against an axis-aligned box and nothing else: not the
orientation, and not the path between two accepted poses. A joint-only command with no `target_pose` is
accepted. A margin that inverts an axis raises at construction, so that failure lands at boot.

`robot.safety.limits.enforce` is read by nothing. It is in the schema and in `robot.yaml`, but only
`workspace_margin_mm` is read from that block: the workspace guard is always wired, and only the other
families consult an `enforce`. Setting it false does not disable the workspace guard. To widen the box,
edit `robot.workspace_limits`; to drop the margin, set `workspace_margin_mm: 0.0`.

**Joint limit** resolves its envelope in order: static `min_deg` and `max_deg` from YAML, then the built-in
UR table keyed on `capabilities.vendor == "ur"` and `.model`, then nothing, which means unavailable and a
rejection. The UR table is the factory envelope, not your installation limits, and no driver reads joint
limits from the controller. The guard can therefore pass a configuration the controller protective-stops
on; the controller is the tighter backstop.

Any cell whose vendor is not `ur` supplies `min_deg` and `max_deg` in YAML. That includes the simulator,
whose capability vendor is `sim`, and every KUKA cell. The `sim` layer supplies the UR factory envelope. Do
not paste those numbers onto a KUKA cell: they are far wider than any KR axis. KUKA per-axis limits come
from the controller or the datasheet, and `config/robot/templates/kuka_eki.yaml` ships no `joint_limits`
block. `create_arm` raises `JointLimitTableMissing` at build for any arm that enforces this guard and
resolves no table, naming both keys. An arm built by hand past the factory still fails closed at run time.

**IK quality** runs five checks: non-finite joints, DoF mismatch, joint jump from `current_joints`, limit
proximity and near-singularity. The singularity test needs an arm advertising `has_native_fk`, and a
failed Jacobian probe returns unavailable, not a pass. Two gaps: with no `current_joints` there is no jump
check, and limit proximity is skipped when no envelope resolves.

**Payload** checks the *declared* envelope; it cannot know what is bolted to the flange. Its checks repeat
the schema validators on purpose: a config assembled outside schema validation, such as a grasping preset
overlay or a hand-built test config, would otherwise reach the controller with a bad payload.

**Motion continuity** bounds three step sizes against the last accepted target. The first command after
`reset()` always passes, and a DoF mismatch skips the joint check. It bounds the net change between two
commanded endpoints and never looks at the path between them; the path gates in 5.4 do.

### 4.5 The two documented skip-sets

Both are code constants, not operator YAML, because a typo in a skip-set would silence a guard.

`gate_joint_target` skips `workspace`, `ik_quality` and `motion_continuity`. A commanded joint move to a
park, home or scan pose is a deliberate restart with no Cartesian target, so only the destination guards
run, and the continuity memory is reset on both sides.

The planner path skips `ik_quality` and `motion_continuity` only. Those two stand in for "a blind
interpolator will not jump", which matters only when a move interpolates in joint space. A planner owns a
continuous, collision-checked path, so a large net joint change such as a shoulder wrap from `+pi` to
`-pi`, about zero physical rotation, is not a jump. On a real UR and in the simulator, a planned move to a
pose is gated with the box on the commanded pose and with joint limits, self-collision and payload on the
configuration the planner returned, and then every sample of the path is gated (5.4).

No guard bounds the middle of a planned path against the Cartesian box, and a commanded joint move meets
no box at all. If the box is what keeps the arm out of a fixture, declare the fixture instead.

### 4.6 `enforce: false` does more than expected

`enforce: false` takes the guard out of the pipeline at construction. It does not merely silence the log:
that family's safety surface is gone until the flag comes back. `omitted_guards` names what was left out
and `guard_names` what runs, so print them (4.1) rather than assume.

### 4.7 The two silent-hole keys

`self_collision.kinematics_model` looks optional and carries weight. Unset on an arm whose vendor is not
`ur`, such as the simulator (physically a UR5e), the arm capsules resolve to `None`. The guard then checks
only base, tool and fixtures, with no arm against arm and no error. The exact-mesh path has the same gate.

`self_collision.kinematics_base_yaw_deg` looks cosmetic at its 0.0 default. The bundled UR DH base is
rotated 180 degrees about Z from the simulator's UR5e `base_link`. The reconcile is a pure base rotation,
so every distance between links is preserved: arm against arm still looks right while every fixture check
is wrong ([safety-math.md](../safety-math.md)). The `sim` layer sets both keys; the base tree leaves them
unset:

```powershell
python -m src.config explain robot.safety.self_collision.kinematics_model --profile sim
```

### 4.8 Cross-validators that fire at config load

[`robot_schema.py`](../../src/config/schema/robot/robot_schema.py) refuses three shapes of half-migrated
cell. On an enabled simulation cell, `self_collision.kinematics_model` must equal `sim.robot_model`; on
`vendor: ur` it must equal `ur.model`; and setting it on a vendor that is neither `ur` nor `sim` is an
error. Profiles compose, so without these rules a chain that put a simulator layer onto a KUKA cell would
load with `kinematics_model: ur5e`, and the guard would judge a KUKA arm by UR5e link lengths.

---

## 5. Self-collision: two backends, and what each is worth

### 5.1 The two backends

The config surface is one block: `python -m src.config where self_collision --tier all --limit 500`.

**`capsule`** needs nothing installed: a chain of capsules (default link radius 60 mm), a base column and a
tool capsule at the commanded TCP, in closed form and under a millisecond
([safety-math.md](../safety-math.md)). `tool_model: capsule`, the default, is a bounding cylinder along the
approach axis with radius `tool_radius_mm`, default 70.0. Against a bin wall it is too conservative: a
two-finger gripper in a tote is about 27 mm wide across its closing axis, while a 70 mm radius claims
140 mm in every direction, so an off-centre target is falsely refused at the wall. `tool_model: finger` is
a thin capsule along the grasp's closing axis. The capsule backend also skips every link pair within two
indices, the wrist cluster, because capsules with realistic radii always overlap there.

**`fcl`** is exact mesh distance against committed per-link meshes, placed by the DH chain and checked
against the simulator's own USD. Pairs within one DH frame are skipped and every farther pair is checked
exactly, so the wrist pairs the capsule backend skips are covered, and each link is also checked against the
declared fixtures. Coal is preferred and `python-fcl` is the fallback. Both run the same meshes, pairs,
thresholds and distance query, so the choice changes speed, not verdicts.

**`fcl` is the default.** The capsule proxy roots its tool capsule in `target_pose`, and a commanded joint
move or a sample of a path carries none, so there the proxy does not model the gripper at all. The two
backends disagree on a substantial fraction of random configurations, always in one direction: the proxy
accepts configurations whose true mesh distance is zero. The mesh check costs about two orders of magnitude
more per check, which is still small against a move.

A wrist camera is part of the arm. A rig that declares `camera.cameras.rigs[<id>].body` names a camera
model from `config/cameras/` (its housing as a box in the colour optical frame), a `margin_mm` and a
`bracket`, and the rig's eye-in-hand calibration places it on the flange. On a cell whose planner is cuRobo
or whose guard reads hand geometry, the planner loads it as the link `wrist_camera_<rig>`, and the exact
guard holds it on frame 6 and checks it against `wrist_2`, every farther link and every fixture. A fixed
camera does not register the housing as an obstacle, and a carried part stays checked against it. No
evidence file is measured per camera: a proof, run at the desk and when the planner starts, shows that the
sphere fill holds every point of every grown box. Such a cell refuses an enabled eye-in-hand rig without a
body, a body its calibration cannot place, and a camera model the registry does not hold. The Isaac wrist
camera is not a rig and carries no body.

### 5.2 Does this robot actually have exact-mesh authority?

Bundles are per robot, `{model}_collision_meshes.npz`, and a ur5e bundle says nothing about a UR3e cell.
They ship in [`src/robot/safety/data/`](../../src/robot/safety/data/), one per arm, and that directory is
the list: a new model gets exact meshes as soon as its bundle lands there, with no code change. The hand a
cell names is a bundle of its own, `{hand}_hand_meshes.npz`, composed onto the arm when the guard loads and
placed on the flange by the rotation the declared tool frame derives, so one hand file serves every arm.

Ask `mesh_backend_status(model, None, hand)` from `src.robot.safety._fcl_self_collision` for your arm and
hand. The tokens are `ok`, `unknown_model`, `no_bundle`, `no_hand_bundle`, `hand_bundle_refused` and
`no_engine`, and `tests/test_status_tokens_are_documented.py` fails when this paragraph and the guard
disagree in either direction. `unknown_model` means no bundled DH row exists for that key, so an arm that
is not a UR has no exact-mesh authority. `no_bundle` means the arm has no bundle yet. `no_hand_bundle`
means the named hand has no bundle of its own, and `hand_bundle_refused` that it has one the guard cannot
place: a part other than the gripper and its two fingers, a missing or malformed array, or fingers that do
not lie along the model's +Y. `no_engine` means neither Coal nor `python-fcl` imports.

A missing arm bundle is baked with
[`scripts/isaac/bake_ur_meshes_from_urdf.py`](../../scripts/isaac/bake_ur_meshes_from_urdf.py) in the plain
environment, with no simulator and no GPU. It reads the Universal Robots collision STL files, pinned to one
upstream commit, and places them through a URDF rendered from the vendored Universal Robots config, so a
bundle follows from pinned bytes. Run it without `--write` first: it compares the bake with the committed
bundle for that model and refuses to write a difference nobody named. Every arm bundle here was baked that
way; against the copies a simulator ships, four links of the family differ by 0.5 to 2.0 mm and `ur10` by
up to 65 mm.

```powershell
python scripts/isaac/bake_ur_meshes_from_urdf.py ur10            # compare with the committed bundle
python scripts/isaac/bake_ur_meshes_from_urdf.py ur10 --write    # write, when the comparison agrees
```

Which arms a hand may be composed onto is not a token. It is measured, one committed evidence file per
combination under `src/robot/safety/planning/robot/evidence/`, and a cuRobo cell starts no planner on a
combination without one.

### 5.3 Asking for `fcl` does not guarantee getting it

**This guard falls back rather than failing closed.** When the engine or the model's bundle is missing, it
logs one warning naming the reason and runs the capsule proxy. The guard stays on, weaker, and says so. A
cell that is not a UR has no bundle and always runs the proxy. No bundle carries base geometry, so while
the mesh path runs, the declared pedestal column (`base_radius_mm`, `base_height_mm`) is not modelled. The
`SelfCollisionSafetyConfig` docstring says an `fcl` guard without meshes rejects; the guard falls back as
described here.

Check what resolved with `python -m src.robot.safety.planning --check`, and on a real cell the checklist's
`exact mesh engine` row (section 7).

**Do not take a capsule verdict as evidence about a real cell.** Keep `backend: fcl`, set
`kinematics_model` and `kinematics_base_yaw_deg` for your arm (4.7), consider `tool_model: finger`, add the
bin walls as `fixtures`, and set `planner_margin_mm` (section 6). If `mesh_backend_status` for your arm and
hand is not `ok`, treat the cell as having no self-collision authority.

### 5.4 The two path gates

`gate_joint_path` and `gate_planned_path` judge every configuration of a path before any of it is
commanded, with the guards a joint move gets: joint limits, self-collision including fixtures, and payload.
No key switches them off and there is no stride to set: a path that grazes a fixture halfway and lands
clear is exactly what an endpoint check misses. The step comes from the collision margin, the coarsest
sampling the check can survive. The reach comes from the arm and what its flange carries: the hand's
sphere map as the guard places it, a wrist camera's fill and a declared carried part. For the last joint,
which turns about the tool axis, a carried point counts by its distance from that axis. A robot or a hand
whose reach does not derive is refused rather than sampled by a guessed number.

[`src/robot/safety/continuous_monitor.py`](../../src/robot/safety/continuous_monitor.py) runs the
exact-mesh backend over every interpolation waypoint of a move, with its own clearance margin and its own
fail-safe: a check that overruns its budget or cannot run returns a stop or a hold, never a continuation.
`ContinuousGuardProfile` defaults to `enabled=False`, `margin_mm=8.0` and `max_check_ms=12.0`, and no
pipeline includes it by default; the simulation runners wire it through
`wire_safety_guards(arm, continuous_guard=True, ...)`. The margin has a ceiling: a natural grasp puts the
arm's own `wrist_1` and `wrist_3` pair at about 19.6 mm, so a margin above that stops a good pick.

---

## 6. The motion-planning requirement

Read this before you report any pick rate. The safety argument is that motion is collision-planned by
cuRobo and collision-checked against exact meshes by Coal or `python-fcl`. A missing engine does not weaken
that argument; it replaces it. Blind IK proposes self-colliding branches that the guard then correctly
rejects, which looks like bad grasping, and the capsule proxy accepts what the mesh check refuses.

Neither engine can be a line in `requirements.txt`. cuRobo needs its own Python 3.10 interpreter, because
one process holds one `warp` and the simulator ships a version cuRobo cannot use. Coal has no Windows wheel,
which is why `python-fcl` is the fallback that ships. Install both from
[`ext_deps/README.md`](../../ext_deps/README.md), the page the failure banner names;
`scripts/ext_deps/install.ps1` puts them where the defaults look. Then probe:

```powershell
python -m src.robot.safety.planning --check      # paths only, milliseconds
python -m src.robot.safety.planning --doctor     # loads both engines, seconds
python -m src.robot.safety.planning --check --model ur3e --hand robotiq_hande
```

`--check` reads paths, so it can report available while an operating-system policy refuses one of the
sidecar's libraries, and the only symptom is a collapsed pick rate. `--doctor` loads the engines and exits
2 when an application-control policy is the cause ([docs/code-integrity.md](../code-integrity.md)).
`--check` exits 0 when the stack is fully anchored and 1 otherwise; on a machine with no GPU environment, or
for a cell that names no hand, 1 is the expected reading. Exit 0 does not prove cuRobo runs, because the
check only verifies that the environment's interpreter exists. `--model` and `--hand` probe another arm
and hand than the config names.

**A cuRobo cell refuses motion rather than degrading.** `robot.ur.motion_planner` is `ik` or `curobo`,
shipped `curobo`. With cuRobo unavailable, a real UR's move returns `CONTROLLER_REJECTED`, and no
collision-free plan returns `TIMEOUT`; it never falls back to blind IK. A cuRobo simulator arm that cannot
reach its sidecar refuses its motions too, and `resolve_runner_planner` refuses the run one layer up. The
simulator's planner is a runner flag, not a config key. `--motion-planner ik`, on the runners that take it,
says a run is deliberately unplanned, and a pick on such a run is refused before the jaws open: ik keeps no
straight line, and the pick's descent is one.

**A cuRobo motion needs a camera world or a decline.** Every verb of a cuRobo UR arm, and of a cuRobo
simulator arm outside mock mode, first asks whether a live camera world is wired or the caller declined one.
With neither, it refuses before planning: status `UNSUPPORTED`, stamp `MISSING`, and a message that starts
`Refused before planning: this cell plans with cuRobo, no live camera world is wired to this arm`.
`move_joint` and `move_linear` raise `RobotMotionRejected` with that result, and `move_home` returns False.
On a `Robot`, a decline is `decline="<why>"` on a verb or a `with robot.without_camera_world("<why>"):`
block. On a bare arm it is `camera_world=CameraWorldDecline("<why>")` on `move` or `move_to_joints`, or
`with arm.without_camera_world("<why>"):`. The world itself is `safety.planning_world` with a calibrated
RGB-D rig ([05](05-pick-loop.md)). A cuRobo cell whose calibrated rig gives no world is refused at the
build, and the checklist's `camera world` row blocks first. Every simulator runner states its choice at
boot (`bootstrap_sim_cell(camera_world=...)`), and a cuRobo boot that states nothing is refused before the
arm is built.

**The simulator's boot gate.** `require_motion_stack` in
[`src/willy_sim/harness/bootstrap.py`](../../src/willy_sim/harness/bootstrap.py) runs before the simulator
boots, keyed on the cell's descriptor and `kinematics_model`, and raises `DegradedMotionStackError` unless
`WILLY_ALLOW_DEGRADED_MOTION=1`. It prints a box rather than a log line, because a simulator boot prints
thousands of lines, and the box states the consequence: a pick rate measured now is not the rate of the
configured system. A cell whose `self_collision.backend` is not `fcl` is not held to the mesh engine, and
one that asks for `ik` is not held to cuRobo. Try it without a simulator; it returns quietly on a machine
with both engines and raises with the box on one without:

```powershell
python -c "from src.willy_sim.harness.bootstrap import require_motion_stack; require_motion_stack('curobo', robot_config='willy_ur5e.yml', kinematics_model='ur5e', exact_mesh_collision=True)"
```

**The planner margin.** cuRobo plans against spheres and the guard re-checks exact meshes. Nothing tells
the planner about `min_distance_mm`, so without a margin it keeps returning configurations just inside the
guard's threshold, the guard refuses them, and that reads as bad grasping. `apply_self_collision_margin` in
[`src/robot/safety/planning/_curobo_margin.py`](../../src/robot/safety/planning/_curobo_margin.py) adds half
the margin to each link's self-collision buffer, because the planner subtracts both links' buffers from a
pair distance.

Measure `planner_margin_mm`; do not copy it. The UR family was measured at 4 mm: against the fitted sphere
map a UR5 finds a retract at 4 and 6 mm and no pose at all at 8 or 10 mm, and a thinner-linked arm reads as
permanently self-colliding once every sphere is inflated for a larger arm. `robot.sim.yaml` and
`robot.ur3e.yaml` declare 4.0 with that reasoning in their comments. `robot.ur5e.yaml` leaves the key
undeclared on purpose: a UR cell that plans with cuRobo and declares no margin refuses to start its planner,
and a committed evidence file must have measured its combination at the margin it declares. Spheres are not
meshes, so the buffer is a cushion, not a proof; the guard, never the planner, decides what executes. The
wire contract and the UR waypoint execution are in
[`src/robot/safety/planning/README.md`](../../src/robot/safety/planning/README.md), and the client and the
UR execution logic are exercised with fakes.

---

## 7. Bring-up, and where to look when it goes wrong

`python -m src.robot.execution.real_cell --check` is the desk checklist, and it touches nothing. From Python
it is `Cell.from_tree(load_tree()).preflight()` or `robot.preflight()`. Against the shipped base tree as a
UR cell it blocks on the items below. That is the correct state of a freshly configured real cell, not a
defect:

| Blocking item | What it looks like at the bench if you skip it |
|---|---|
| `gripper.tool_frame.source` is `undeclared` | a top-down grasp at z = 37 mm drives the flange there, and the fingertips through the bench |
| `safety.payload` has `enforce: true` and `mass_kg: 0.0` | `connect()` refuses it; pushed, a zero payload leaves the controller's stop model reading low |
| `safety.planning_world.payload.length_mm` is undeclared | every lift after a grasp is planned as if the hand were empty; declare the overhang, or `enabled: false` |
| no `CAMERA->BASE` transform on the primary rig | grasps stay in the camera frame and every motion is rejected as `INVALID_TARGET`: a cell that looks hung |
| no live camera world for the planner | every pick motion is refused as `UNSUPPORTED` before the arm moves, naming the missing world |
| `safety.self_collision.planner_margin_mm` is undeclared | undeclared is not zero: the planner never starts, and the first planned move is `CONTROLLER_REJECTED` |
| `gripper.model` is unset | the exact-mesh guard checks the hand this key names, and the base tree names none: the build is refused |

A machine without the cuRobo environment or without Coal also blocks on `cuRobo environment` and
`exact mesh engine`. Those are facts about the machine the checklist runs on, and the `ext_deps` install
clears them.

[`config/robot/robot.ur5e.yaml`](../../config/robot/robot.ur5e.yaml) is the worked example for a real
bench. It leaves the tool frame, the payload, the camera-to-base calibration and the camera itself unset on
purpose, and carries each as a commented block that says what to measure. Its rule for which keys carry a
value: a wrong value that fails closed ships with its assumption stated, so a workspace box that is too
small refuses a motion visibly; a wrong value that fails open does not ship, because a plausible tool frame
or payload drives the arm into the bench and logs a success.

A blocking item does not always stop the connect. A cell with a blocking checklist can connect and then
refuse every motion, which reads as a broken robot. What refuses the connect is the driver's own preflight,
not this checklist. Warnings never block, and the bench rows are questions no interface can answer.

The order that works:

1. `python -m src.robot.drivers.doctor --require ur`: the SDK is installed. Exit 0 is the pass.
2. `python -m src.robot.execution.real_cell --check`: fix everything blocking.
3. `python -m src.robot.safety.planning --doctor`, then `python -m src.robot.execution.real_cell --start-planner`:
   the planner environment, and this cell's planner starting, before any motion.
4. `python -m src.robot.perception --prompt "..."`: the camera and the models, with no robot.
5. Calibrate each camera, one command per rig ([03](03-calibration.md)), then `real_cell --dry-run`, then
   `--runs 1`, then a campaign.

The step-by-step procedures are [cell_bringup.md](../runbooks/cell_bringup.md) and
[real_cell_first_pick.md](../runbooks/real_cell_first_pick.md); the same steps from Python are
[`examples/real_robot/`](../../examples/real_robot/).

Along the way: confirm the mesh bundle for this robot (5.2); keep the exact-mesh guard on and tell it which
kinematics it guards (5.3); measure `planner_margin_mm` on this robot (6); print the guard pipeline (4.1),
since a missing guard means an `enforce: false` removed that safety surface; and for any vendor that is not
`ur`, supply static joint limits (4.4). Re-measure after any of these, because margins, the yaw reconcile
and the planner all change which motions are proposed and which are accepted.

Logs land in `logs/robot/`, with a file per module under `logs/robot/modules/`. The self-collision guard and
the simulator driver's planner resolution log on the plain module logger, so read the runner's standard
output too. Four symptoms cover most of the confusion:

- `JointLimitTableMissing` at build, or every move `CONTROLLER_REJECTED` on a KUKA, simulator or
  hand-built arm: the joint-limit guard has no table (4.4).
- Picks that look like bad grasping on a cell that runs `ik`: blind IK proposing self-colliding branches
  that the guard correctly rejects (6).
- Collisions slipping through on `backend: fcl`: the engine or the bundle is missing and the capsule proxy
  runs with its wrist skip (5.3).
- A connect refused with `NoRealGripper` for `vacuum` or `jaw_io`: the arm does not advertise
  `SupportsDigitalIO` (section 3). A digital-I/O `connect()` that raised with no I/O: the gripper was
  connected before the arm.

---

## 8. What this layer does not give you

This is a software collision-avoidance layer, not the safety system of a cell. It reduces collision risk.
The safety guarantee of a real cell is independent, certified functional safety: the vendor's safety-rated
stop, hardware emergency-stop circuits, and ISO 10218, ISO/TS 15066 and ISO 13849, all running
independently of this software. A green simulation result is not a commercial safety claim. The planning
README and [safety-math.md](../safety-math.md) say the same.

| Area | Evidence |
|---|---|
| The preflight on every pick, and the exact-mesh backend of the `sim` layer | measured in simulation |
| The planner gated by the mesh guard, its margin, and the motion-stack gate with its banner | measured in simulation |
| The simulator's jaw and suction grippers with their profiles | measured in simulation |
| The UR driver: connect, power, motion, digital I/O, tool frame, payload, protective stop | measured against real controller software |
| A UR executing this driver or a planned motion on a physical arm | never touched hardware |
| KUKA, which also has no built-in joint-limit table and no bundled DH chain, so no exact-mesh authority | never touched hardware |
| The wiring of both digital-I/O grippers | never touched hardware |

Next: [05](05-pick-loop.md), what happens above this layer once a motion is allowed, and
[06](06-grippers.md), making an end-effector move.
