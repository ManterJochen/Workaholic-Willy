# Dummy arm (`src/robot/drivers/dummy`)

An in-memory arm for a desk: it keeps a pose and a joint vector, commands nothing and has no
kinematics. The desk profile `console_dummy` builds it, so the library's calls, the simulation examples
and the rehearsal run with no robot, no SDK and no GPU.

```python
from willy import Pose, Robot, load_tree

robot = Robot.from_tree(load_tree("console_dummy"))   # a dummy arm and a dummy hand
with robot.connected(), robot.without_camera_world("desk run, no camera"):
    print(robot.home())
    print(robot.move(Pose.tool_down(450.0, 100.0, 300.0)))
    print(robot.arm.get_tcp_pose())      # the pose it was sent, and nothing else
```

The same calls with the hand, a pick and a place are in
[examples/simulation/02_a_robot_at_the_desk.py](../../../../examples/simulation/02_a_robot_at_the_desk.py).
This package has no command line.

## The nouns

| Noun | Built by | Verb | Returns |
|---|---|---|---|
| `DummyRobotArm` | `Robot.from_tree(tree)` on a `dummy` cell, `create_arm("dummy", dof=6)`, or `DummyRobotArm(dof=6)` | `move(pose)`, `move_linear`, `move_to_joints`, `get_tcp_pose()` | `MotionResult`; the last commanded pose |
| `DUMMY_CAPABILITIES` | a constant | | vendor `dummy`, model `dummy-6dof`, no native FK or IK, `is_simulated=True` |

`DummyRobotArm(*, dof=6, initial_pose=None)` starts at `(400, 0, 300)` mm with an identity rotation;
`initial_pose` must be in `Frame.BASE`. A `dof` other than 6 gives a `dummy-{dof}dof` model. A `dummy`
cell builds the arm with these defaults: the `robot.dummy` block carries no fields.

## What it refuses

| Refusal | When | What to do |
|---|---|---|
| `INVALID_TARGET` from `move` | a pose not in `Frame.BASE` | Tag the pose with the base frame |
| `CONNECTION_ERROR` from `move` | a move before `connect()` | Connect first |
| a typed error from `move_to`, `move_linear` and the reads | the same two faults; these check the connection before the frame | The same |
| `TypeError` | `model=` passed to the constructor | The model follows from `dof` |

A joint vector whose length is not the arm's `dof` is refused too. `register` is accepted for Protocol
compatibility and ignored, and so is a `config=` handed to `create_arm`.

## What it does not do

| | |
|---|---|
| Kinematics | `fk` returns the last commanded pose and `ik` the last commanded joints: the right shapes, not physical values |
| A workspace, collisions, latency | any base-frame pose is accepted, every call returns at once, `wait_until_steady` returns immediately |
| Safety gating | it carries no preflight; `robot.safety()` reports `UNGATED` and `robot.route()` says nothing real moves |
| Physics | for picks with contact and kinematics, use the [Isaac arm](../sim/README.md) |

A green run on this arm says the calls and the library agree; it says nothing about grasping.

## Status

| Capability | Evidence |
|---|---|
| The whole `RobotArm` Protocol, in memory | measured in simulation: every desk example and the rehearsal run on it |

## Files

| File | Holds |
|---|---|
| `arm.py` | `DummyRobotArm` and `DUMMY_CAPABILITIES` |
| `__init__.py` | re-exports both |

## Details

- [drivers](../README.md): the registry, the doctor and the other vendors
- [grippers](../../grippers/README.md): `DummyGripper`, the hand that pairs with this arm
- [config/robot/robot.console_dummy.yaml](../../../../config/robot/robot.console_dummy.yaml): the desk profile
