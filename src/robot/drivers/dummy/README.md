# Dummy arm driver

A pure-Python, dependency-free `RobotArm` for tests and offline development. No robot SDK, no GPU,
no physics, no kinematics.

It is an in-memory implementation of the vendor-neutral [`RobotArm`](../../core/robot_arm.py)
Protocol, so the whole pick stack, meaning the perception adapters, the safety guards, grasp
execution and the calibration routines, runs with no hardware at all. Select it with
`RobotVendor.DUMMY`.

This is the arm only. The matching in-memory gripper, `DummyGripper`, lives in
[`grippers/dummy.py`](../../grippers/dummy.py) and is selected with `GripperVendor.DUMMY`.

## Contents

| File | Role |
| --- | --- |
| [`arm.py`](arm.py) | The entire driver: the `DUMMY_CAPABILITIES` constant and the `DummyRobotArm` class. |
| [`__init__.py`](__init__.py) | Re-exports `DummyRobotArm` and `DUMMY_CAPABILITIES`. |

One self-contained file by design. There is no `connection.py`, no `config.py` and no vendor
submodule.

## Public surface

`DummyRobotArm(*, dof=6, initial_pose=None)` holds `(joints, tcp_pose)` in memory with no kinematic
model. `initial_pose` must be in `Frame.BASE`; the default TCP is `(400, 0, 300)` mm with an
identity XYZW quaternion, labelled `dummy-home`.

`DUMMY_CAPABILITIES` is a fixed `RobotCapabilities`: `vendor="dummy"`, `model="dummy-6dof"`,
`dof=6`, joint and linear move supported, no async move, no native FK or IK, no force control, and
`is_simulated=True`. Constructing with a `dof` other than 6 derives an equivalent `dummy-{dof}dof`
profile.

The full `RobotArm` Protocol surface is implemented: the lifecycle (`connect` and `disconnect`, both
idempotent), state (`get_tcp_pose`, `get_joint_positions`), motion (`move_joint`, `move_linear`,
`move_to_joints`, `stop`), stubbed kinematics (`fk`, `ik`), and the high-level helpers
(`is_inside_workspace`, `move_to`, `move_home`, `wait_until_steady`, and the typed `move`).

## Usage

```python
from src.robot.core import RobotVendor
from src.robot.drivers import create_arm

arm = create_arm(RobotVendor.DUMMY)   # dof=6, TCP at (400, 0, 300), identity rotation

# Or directly:
from src.robot.drivers.dummy import DummyRobotArm
arm = DummyRobotArm(dof=6)

arm.connect()
arm.move_home()
pose = arm.get_tcp_pose()             # Frame.BASE, millimetres, XYZW
```

There is no `python -m` entry point. The dummy is constructed in-process by tests and offline
scripts.

The invariants that need no controller are enforced for real: `Frame.BASE`-only poses, matching
degrees of freedom, and a must-be-connected guard, each raising a typed error. That is what makes
this worth testing against instead of a mock object.

## What it deliberately does not do

| | |
| --- | --- |
| No real kinematics | `fk` returns the last commanded TCP pose and `ik` the last commanded joints. The shapes are correct and the values are not physically meaningful. Use the UR or the sim driver for anything kinematic. |
| No workspace box, no collision model, no latency | `is_inside_workspace` accepts any `Frame.BASE` pose, calls return synchronously, and `wait_until_steady` returns immediately. The safety package's geometric guards run around this driver, not inside it. |
| Not a physics simulation | For simulation-based validation, meaning known-pose and vision picks and hand-eye calibration, use the [sim](../sim/README.md) Isaac driver. |

## Two sharp edges

**The typed `move` and the boolean `move_to` check in a different order.** On success both record the
pose. `move` reports a fault as a typed failure instead of raising: `INVALID_TARGET` for a non-BASE
frame, `CONNECTION_ERROR` when not connected, otherwise an executed `MOVE_TO`. It checks the frame
before the connection, whereas `move_to` and `move_linear` check the connection first.

**There is no `model` keyword argument.** `create_arm(RobotVendor.DUMMY, ...)` pops and discards any
`config=` keyword, so it accepts only `dof` and `initial_pose`. `DummyRobotArm.__init__` has no
`model` parameter and passing one raises `TypeError`; the capability `model` string is derived from
`dof` alone. `register` is likewise accepted for Protocol compatibility and ignored.

## See also

- [drivers](../README.md), the vendor-neutral driver registry and `create_arm`
- [sim](../sim/README.md), the Isaac physics-simulation arm with real kinematics and collisions
- [ur](../ur/README.md), the Universal Robots RTDE driver
- [grippers](../../grippers/README.md), the end-effectors, including the matching `DummyGripper`
