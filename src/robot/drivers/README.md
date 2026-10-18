# Arm drivers (`src/robot/drivers`)

One driver per arm vendor, each behind the `RobotArm` Protocol: UR over RTDE, the Isaac Sim arm, and
a dummy arm for a desk. `robot.vendor` in your config picks the driver and `Robot.from_tree` builds it.

```python
from willy import Robot, load_tree
from src.robot.drivers import create_arm

tree = load_tree()                      # the cell WILLY_PROFILE names
robot = Robot.from_tree(tree)           # robot.vendor picks the driver; nothing connects yet
print(robot.arm.capabilities)           # vendor, model, degrees of freedom, what this arm supports

arm = create_arm(tree.robot.vendor, config=tree.robot)   # the arm alone: no hand, no lock
```

Before a real arm is powered, ask this machine whether it can drive it. The probe connects nothing:

```bash
python -m src.robot.drivers.doctor               # one row per arm and gripper vendor
python -m src.robot.drivers.doctor --json        # the same rows as JSON
python -m src.robot.drivers.doctor --require ur  # exit 1 unless this machine can drive a UR
```

It exits 0 when the table printed or the required vendor is ready, 1 when the required vendor is not
ready, and 2 for a vendor name it does not know. The calls at a cell are in
[examples/real_robot/03_connect_and_move.py](../../../examples/real_robot/03_connect_and_move.py).

## The nouns

| Noun | Built by | Verb | Returns |
|---|---|---|---|
| `RobotArm`, one class per vendor | `Robot.from_tree(tree)`; `create_arm(vendor, config=tree.robot)` for the arm alone | `move(pose)`, `get_tcp_pose()` | `MotionResult`; a `Pose` in the base frame |
| `ReadinessReport` | `Host.local().readiness()` | `ready_for(vendor)`, `require(vendor)`, `render()` | a bool; `RobotConnectionError` naming what is missing |

## The vendors

| `robot.vendor` | Driver | `create_arm` takes | Needs on this machine |
|---|---|---|---|
| `ur` | `URRobotArm`, [ur/](ur/README.md) | `config=` the robot section | `ur_rtde`, imported at `connect()` |
| `sim` | `IsaacRobotArm`, [sim/](sim/README.md) | `config=` a `SimRobotConfig`, which `Robot` builds from `robot.sim` | Isaac Sim at `connect()`, or `mock_mode` |
| `dummy` | `DummyRobotArm`, [dummy/](dummy/README.md) | optional `dof` and `initial_pose`; a `config=` is discarded | nothing |

At this boundary poses are millimetres and XYZW quaternions tagged `Frame.BASE`, and joints are
radians. A vendor's own units (UR axis-angle metres) stay inside its
subpackage, and so does its SDK import: importing this package works on a machine with neither
`ur_rtde` nor Isaac installed.

The UR and the Isaac arm both plan with cuRobo by default, and neither falls back: where the cuRobo
planner cannot start, every planned motion is refused as `CONTROLLER_REJECTED` rather than run on plain
IK, which sees nothing in the cell. Each driver's README says what else it refuses.

## What it refuses

| Refusal | When | What to do |
|---|---|---|
| `RobotConnectionError`, host not ready | `Robot.from_tree` on a vendor whose SDK does not import here | Run the doctor; install `ur_rtde` from `requirements.txt` |
| `ValueError`, unknown robot vendor | `robot.vendor` names no driver (`kuka`, `franka` and `ros2` were removed on 2026-10-17) | Use `ur`, `sim` or `dummy` |
| `JointLimitTableMissing` | the arm's joint-limit guard resolves no per-axis table: the arm registry describes no arm of the model it reports | Describe the arm (`config/arms/<model>.yaml`, [add_an_arm_model.md](../../../docs/runbooks/add_an_arm_model.md)) |
| `ValueError` | a vendor name that is not a `RobotVendor` | Use one of the names above |
| `TypeError` | no `config=`, an unexpected keyword, or a sim config that is not a `SimRobotConfig` | Pass what the vendor table names |

`create_arm` asks the joint-limit question of every arm it hands back, because an enforcing guard
with no table would refuse every motion as `controller_rejected` and send you to the controller for a
description missing from the arm registry. Only an arm reporting a model the arm registry describes, every UR and
the Isaac arm, carries a table; no tree writes one of its own since 2026-10-17, and no tree switches the guard off.
An arm that gates nothing, as the dummy does, passes this check; `robot.safety()` reports it.

## Host readiness

`ready` has one definition: the vendor has a registered driver and every SDK module it needs imports.
The printed table and the gate `Robot.from_tree` runs read the same rows, so they cannot disagree.

```python
from src.robot.drivers.host import Host

report = Host.local().readiness()
print(report.render())                  # one row per arm and gripper vendor
report.require("ur")                    # raises RobotConnectionError naming what is missing
```

`mock_mode=True` waives the SDK half for `sim` and `dummy` only, the two vendors that drive no device;
it never waives a missing driver. The probe imports each thin client library, because a package the
operating system refuses to load is found only by importing it, and looks Isaac up by name only,
because importing Isaac starts a renderer. A module that is installed and will not import is covered in
[docs/code-integrity.md](../../../docs/code-integrity.md).

## Status

| Driver | Evidence |
|---|---|
| UR over RTDE | run on a physical cell: a UR10 (CB3), connect, moves, cuRobo plans and home; measured against real controller software in URSim, see [ur/](ur/README.md) |
| Isaac Sim arm | measured in simulation: the picks and calibrations in [willy_sim](../../willy_sim/README.md) |
| Dummy arm | measured in simulation: every desk example and the rehearsal run on it |

The UR driver is the one in this package that has moved a physical arm.

## Adding a vendor

1. Add the name to `RobotVendor` in [`src/robot/core/vendor.py`](../core/vendor.py).
2. Create `src/robot/drivers/<name>/` with an `arm.py` holding the `RobotArm` implementation and a
   capability descriptor. Keep every vendor SDK import inside that package.
3. Register a factory in this package's `__init__.py` with `@register_arm_driver(RobotVendor.X)`,
   importing the SDK inside the function body.
4. If the driver needs a third-party SDK, list its modules in `_ARM_VENDOR_SDKS` in `doctor.py` so the
   readiness probe sees them. Whether the vendor is registered is read from the registry either way.
5. Give the arm a joint-limit table, or document the two YAML keys, because `create_arm` refuses an
   enforcing guard without one. Add tests for Protocol conformance, safety wiring and the typed motion
   result.

Nothing above this package changes: everything past this boundary talks to the Protocol.

## Files

| File | Holds |
|---|---|
| `registry.py` | `create_arm`, `register_arm_driver`, `unregister_arm_driver`, `is_vendor_registered`, `available_vendors` |
| `__init__.py` | the three built-in factories, each importing its SDK inside the function body |
| `doctor.py` | the readiness CLI and `require_arm_vendor_ready`, the gate a build runs first |
| `host.py` | `Host` and `ReadinessReport`, the rows the CLI and the gate share |
| `ur/`, `sim/`, `dummy/` | the three drivers |
| `planned/` | `PlannedArm`, the judging and planning half the UR arm stands on (P0e E8), with the cuRobo planner and the planner frame; [planned/](planned/README.md) |

## Details

- [robot/core](../core/README.md): the `RobotArm` and `Gripper` Protocols, `RobotVendor`, `MotionResult`,
  `MotionStatus` and the typed errors
- [grippers](../grippers/README.md): the hand on the arm, and `create_gripper`
- [safety](../safety/README.md): the preflight every motion passes and the cuRobo planning;
  [docs/safety-math.md](../../../docs/safety-math.md) for the formulas
- [docs/runbooks/cell_bringup.md](../../../docs/runbooks/cell_bringup.md): bringing a cell up, vendor by
  vendor; [docs/cli.md](../../../docs/cli.md) for every command line
- [guide 04](../../../docs/guide/04-robot-and-safety.md): declaring the robot and its safety in config
