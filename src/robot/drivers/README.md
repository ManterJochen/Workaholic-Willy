# Robot arm drivers

Vendor-specific `RobotArm` implementations behind one lazy registry and factory: UR, KUKA, Isaac
Sim and a pure-Python dummy, with `franka` and `ros2` as declared but empty slots.

This is the drivers layer of the downward dependency stack
(`config -> geometry -> perception -> robot/core -> drivers -> safety -> grasping`). Each driver
implements `src.robot.core.RobotArm` and is consumed by `safety`, `grasping` and `execution`.
Nothing here imports upward.

## What this package guarantees

**One entry point.** `create_arm(vendor, **kwargs)` returns a `RobotArm` for any registered vendor.
It validates the result structurally against the runtime-checkable Protocol, so a buggy factory
cannot quietly return the wrong shape.

**Import safety.** A vendor SDK is imported inside a factory body, and for UR and the Isaac sim it
is deferred further, to `connect()`. Importing this package on a host with no `ur_rtde` and no
Isaac installed always works.

**One unit and frame contract at the boundary.** Millimetres, XYZW quaternions, frame-tagged `Pose`
with `Frame.BASE` for the TCP, and joints in radians. A vendor's own conventions, UR axis-angle
metres or KUKA degrees and ABC Euler, stay inside that vendor's subpackage.

**An unselectable vendor fails fast.** `create_arm` on a vendor with no registered factory raises
`RobotConnectionError` naming the vendor, rather than importing something that does not exist.

This package does not own grippers. `create_gripper` and the gripper drivers live in the sibling
[`grippers/`](../grippers/README.md) package, including the Isaac `IsaacGripper`.

## Contents

| Path | Role |
| --- | --- |
| `registry.py` | The lock-guarded registry: `create_arm`, `register_arm_driver`, `unregister_arm_driver`, `is_vendor_registered`, `available_vendors`, `ArmDriverFactory`. |
| `__init__.py` | The package facade. It re-exports the registry surface and registers the four built-in factories (`_make_dummy`, `_make_ur`, `_make_kuka`, `_make_sim`) with their SDK imports inside the function bodies. |
| `doctor.py` | Which vendor SDKs import on this host, as a readiness table, plus `require_arm_vendor_ready` and a `python -m` CLI. |
| `host.py` | `Host` and `ReadinessReport`, the library twin of the doctor. One set of rows answers both the table and the gate. |
| [`ur/`](ur/README.md) | UR over RTDE: `URRobotArm`, `UR_CAPABILITIES`, the digital-I/O bench, cuRobo-planned motion. |
| [`kuka/`](kuka/README.md) | KUKA over EthernetKRL: `KukaRobotArm`, `KUKA_CAPABILITIES`, the TCP and XML transport. No vendor SDK on this side. |
| [`sim/`](sim/README.md) | Isaac Sim: `IsaacRobotArm`, `ISAAC_CAPABILITIES`, the session lifecycle, the model registry, the mm and quaternion adapters. |
| [`dummy/`](dummy/README.md) | `DummyRobotArm` and `DUMMY_CAPABILITIES`: an in-memory arm with no kinematics. |
| [`franka/`](franka/README.md), [`ros2/`](ros2/README.md) | Empty package slots. Each is an `__init__.py` with an empty `__all__` and a README. Nothing is registered. |

Driver classes are exported from their own subpackages, not from the package root. The registry
surface and `RobotVendor` are exported from `src.robot.drivers`.

## Registered vendors and factory kwargs

| Vendor (`RobotVendor`) | Factory kwargs | Notes |
| --- | --- | --- |
| `DUMMY` | optional `dof`, optional `initial_pose`; a forwarded `config` is discarded | Pure Python, no SDK. |
| `UR` | `config=RobotConfig`, required | Builds `URRobotArm`. `ur_rtde` is deferred to `connect()`. |
| `KUKA` | `config=RobotConfig`, required | Builds `KukaRobotArm`. No vendor SDK on this side. Measured 2026-09-10 on the shipped `web` profile: with `safety.joint_limits.min_deg` and `max_deg` unset, as base `robot.yaml` ships them, `create_arm` now raises `JointLimitTableMissing` here rather than handing back a cell that answers `controller_rejected` to every move. |
| `SIM` | `config=SimRobotConfig`, required and type-checked | Builds `IsaacRobotArm`. Isaac is deferred to `connect()`. |
| `FRANKA`, `ROS2` | none | Declared in the enum and not registered. `create_arm` raises `RobotConnectionError`. |

Every factory rejects unexpected kwargs with `TypeError` rather than dropping them. The
application-layer factory forwards `config=` for every vendor, so an operator switches arms by
editing config alone; the dummy factory is the one that discards it.

`create_arm` asks one safety question of every arm it hands back: an arm that wires the joint-limit
guard but resolves no per-axis table for its own `capabilities` is refused with
`JointLimitTableMissing`, naming `robot.safety.joint_limits.min_deg` and `max_deg`. Only the UR
models carry a built-in table, so this is the check a new vendor meets first. An arm that reports
`safety_preflight is None`, as the dummy does, or whose operator set `joint_limits.enforce: false`,
is silent here, because those are stated decisions and `SafetyAttestation` is what reports them.

## Usage

```python
from src.robot.core import RobotVendor
from src.robot.drivers import create_arm, available_vendors

available_vendors()                       # ['dummy', 'kuka', 'sim', 'ur']

arm = create_arm(RobotVendor.UR, config=cfg.robot)
arm.connect()
pose = arm.get_tcp_pose()                 # Pose(Frame.BASE), millimetres, XYZW quaternion
arm.disconnect()
```

Check host readiness before selecting a real vendor:

```bash
python -m src.robot.drivers.doctor               # the per-vendor readiness table
python -m src.robot.drivers.doctor --json        # the same rows as JSON
python -m src.robot.drivers.doctor --require ur  # exit non-zero unless UR is ready
```

Exit codes are `0` where the table printed or the required vendor is ready, `1` where a required
vendor is not ready, and `2` for an unknown vendor name.

## Host readiness has one definition

`ready` means the vendor has a registered driver and every SDK module it needs is importable. The
same rows back the printed table and the programmatic gate, so the two cannot disagree:

```python
from src.robot.drivers.host import Host

report = Host.local().readiness()
report.ready_for("ur")                    # bool
report.require("sim", mock_mode=True)     # raises unless ready; mock mode needs no SDK
print(report.render())
```

Two properties are load-bearing. `registered` is read from the arm and gripper registries rather
than from the SDK map, so a vendor that ships a driver and needs no SDK cannot be reported as
having no driver. And `mock_mode=True` waives the SDK requirement only for the vendors that drive
no real device, which are `sim` and `dummy`.

The probe is the cheapest question in the stack and stays that way: no `connect`, no `build`, no
network, just an import-spec lookup and two registry lookups.

## Driver maturity

| Driver | What is actually proven |
| --- | --- |
| UR | Measured against URSim, which is the real UR controller software speaking real RTDE: dashboard power-on and brake release, `moveJ`, RTDE reads of pose, joints, modes and wrench, digital outputs that read back changed, tool-frame verification against the controller's own register, `setPayload` with a connection rollback, and a genuine protective stop projected end to end. No physical UR has ever executed a line of it. |
| Sim (Isaac) | Import-safe everywhere. A `mock_mode=True` pure-Python path runs with no Isaac, and `connect()` without Isaac raises `IsaacNotAvailableError`. On the workstation the kinematics core is real: Lula FK and IK, live TCP and joint reads, and the whole perceive, grasp and execute path including hand-eye calibration through the real calibration routine. It is a validation platform, not production hardware. |
| Dummy | Works, and is intentionally trivial. In-memory `(joints, tcp_pose)` with no kinematics: `fk` returns the last recorded pose and `ik` the last recorded joints. For tests and offline pipeline development. |
| KUKA | Registered and implemented. `KukaRobotArm` speaks EKI and KRL over TCP with XML and needs no vendor SDK, but no motion has ever run against a live KRC4 or KRC5 from this codebase. The controller-side KRL program under `config/robot/templates/kuka/` has to be deployed and validated by an integrator first. |
| Franka, ROS 2 | Empty slots. `create_arm(RobotVendor.FRANKA)` raises `RobotConnectionError`. |

## Traps

**The sim driver is model-selectable, and the config wins.** `SimRobotConfig.robot_model`
(`ur5e` by default, or `ur3e` or `ur10e`) picks the Lula solver config, the Isaac USD and the
cuRobo `{model}.yml` together, through `sim/robot_models.py`. A disagreeing `WILLY_CUROBO_ROBOT`
environment variable is loudly ignored, because planning one cell against another robot's geometry
has no visible symptom. Only `ur5e` is validated on the workstation.

**cuRobo-planned motion is fail-closed on both real and simulated UR, and the two fail
differently.** `ur/curobo_motion.py` plans a global collision-free joint trajectory and executes it
waypoint by waypoint over `moveJ`, with no blind-IK fallback: a missing cuRobo environment is
`CONTROLLER_REJECTED` and no plan found is `TIMEOUT`, so a real cell with no cuRobo environment does
not move at all. `robot.ur.motion_planner` defaults to `curobo`. The Isaac driver's
`SimRobotConfig.motion_planner` also defaults to `curobo` but degrades to the blind `ik` path where
the environment is absent, which is what keeps a machine with no cuRobo green.

**No gripper factory here.** The sim `IsaacGripper` lives in `grippers/sim/` and is constructed
directly, deliberately unregistered so it can never be selected for a real robot.

## Adding a vendor

Drop a sibling subpackage here that exposes one class satisfying the `RobotArm` Protocol, and
register a factory for it:

1. Add the identifier to `RobotVendor` in `src/robot/core/vendor.py`, if it is not there already.
2. Create `src/robot/drivers/<name>/` with an `arm.py` holding the Protocol implementation and a
   capability descriptor, and keep every vendor SDK import inside that package.
3. Register the factory in this package's `__init__.py` with `@register_arm_driver(RobotVendor.X)`,
   importing the SDK inside the function body and never at module top level.
4. If the driver needs a third-party SDK, list its modules in `_ARM_VENDOR_SDKS` in `doctor.py` so
   the readiness probe can see them. A vendor absent from that map is treated as needing no SDK;
   whether it is registered is read from the registry either way.
5. Add tests for Protocol conformance, safety wiring and the typed motion result.

No pipeline, planner or application-layer change is needed: everything above this boundary talks to
the Protocol.

## See also

- [robot package](../README.md), the layer this driver package sits in
- [core/](../core/README.md), the `RobotArm` and `Gripper` Protocols, `RobotVendor`, `MotionResult`
  and `MotionStatus`, and the typed errors these drivers raise
- [grippers/](../grippers/README.md), the sibling gripper drivers and `create_gripper`
- [safety/](../safety/README.md), the fail-closed preflight and the cuRobo planning these drivers
  consume
- [`scripts/ursim/`](../../../scripts/ursim/README.md), bringing up the controller software the UR driver
  was measured against
