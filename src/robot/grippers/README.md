# Grippers (`src/robot/grippers`)

The drivers for the hand on the arm, each behind the `Gripper` Protocol: a Robotiq over its URCap
socket, an OnRobot RG2 or RG6 over Modbus, a jaw or a suction cup over the controller's digital I/O, a
dummy for a desk and an explicit "no gripper", plus the two simulated grippers Isaac uses.
`robot.gripper.vendor` picks the driver and `Robot` builds it, so you rarely call this package directly.

```python
from willy import Robot, load_tree

robot = Robot.from_tree(load_tree())    # robot.gripper.vendor picks the driver
print(robot)                            # the hand that was built, or the stand-in and why
with robot.connected():                 # the arm first, then the hand
    print(robot.grasp(40.0))            # close to 40 mm and read what the hand measured
    print(robot.is_holding())           # HELD, EMPTY or UNMEASURED
    print(robot.release())
```

The walk at a cell is [examples/real_robot/04_open_and_close_the_hand.py](../../../examples/real_robot/04_open_and_close_the_hand.py);
which hand a tree really builds is [examples/offline/config/which_gripper_gets_built.py](../../../examples/offline/config/which_gripper_gets_built.py).
Before the first close of a digital-I/O hand, measure its pins with the UR bench, which moves no arm:

```bash
python -m src.robot.drivers.ur --read                          # every pin on the tool bank, changes nothing
python -m src.robot.drivers.ur --measure 4=1 --watch 0 --yes   # drive a pin and time the answer
```

## The drivers

| `robot.gripper.vendor` | Driver | Speaks | Needs |
|---|---|---|---|
| `robotiq` | `GripperController` | ASCII on TCP port 63352, opened by the URCap on the UR controller | a UR arm with the Robotiq URCap |
| `onrobot` | `OnRobotGripper` | Modbus TCP to the Compute Box, port 502 by default | the box on your network; RG2 or RG6 only |
| `jaw_io` | `JawIOGripper` | one or two output pins, optional reed switches or a part sensor | an arm with digital I/O: the UR driver |
| `vacuum` | `VacuumGripper` | an ejector pin, an optional blow-off pin and vacuum switch | an arm with digital I/O: the UR driver |
| `dummy` | `DummyGripper` | nothing: a width in memory, clamped to range | nothing |
| `none` | `NullGripper` | nothing | nothing |

None of them needs a third-party package: `robotiq_socket.py` and `onrobot_modbus.py` speak their wire
formats directly. `franka_hand` and `schunk` are reserved names with no driver. Units differ at the wire
and are converted inside each driver: Robotiq positions are counts from 0 to 255, and its speed and
force are fractions from 0 to 1, not newtons; an OnRobot width is an opening in tenths of a millimetre
(larger is more open), its force is in newtons, and it has no activation stroke and no speed register.
Every `jaw_io` and `vacuum` wiring number is config (`robot.gripper.jaw_io.*`, `robot.gripper.vacuum.*`),
so bring-up is measuring, not coding.

`robot.gripper.model` is a separate key: the hand's name in the registry under `config/grippers/`
(`robotiq_2f85`, `robotiq_hande`, `schunk_egu50`), whose geometry the collision guard and the planner
read. [Your own gripper](../../../docs/runbooks/your_own_gripper.md) adds one.

## The nouns

| Noun | Built by | Verb | Returns |
|---|---|---|---|
| `Gripper`, one class per vendor | `Robot.from_tree(tree)`, or `create_gripper(vendor, **kwargs)` | through `Robot`: `grasp(width_mm)`, `release()`, `is_holding()` | `HandReport`; `HoldEvidence` |
| `GripperSubstitution` | set on a `NullGripper` that stands in for a hand that could not be built | | the reason, the vendor asked for, a sentence, the fix |

Offline, `create_gripper(GripperVendor.DUMMY, max_width_mm=85.0)` from `src.robot.grippers` builds a
hand alone, and `connect()`, `activate()`, `set_width_mm(40.0)` and `get_width_mm()` drive it.

## What it refuses

| Refusal | When | What to do |
|---|---|---|
| a `NullGripper` stand-in, then `NoRealGripper` at connect | the tree names a hand this arm cannot drive, one of the cases below | Read the fix the refusal carries |
| `ValueError` from `create_gripper` | a name that is not a `GripperVendor` | The message lists the buildable and the reserved names |
| `RobotConnectionError` from `create_gripper` | `franka_hand` or `schunk` | Use a vendor with a driver; a Schunk on tool I/O works as `jaw_io` |
| `TypeError` from `create_gripper` | a factory returned something that is not a `Gripper` | A bug in a registered factory |

A stand-in is built for an unknown vendor, for `robotiq` on an arm that is not a UR or exposes no
address, for `jaw_io` or `vacuum` on an arm without digital I/O, and for `franka_hand` or `schunk`. Its
`substitution` carries the reason, the vendor asked for and the fix. A `NullGripper` with no
substitution is a cell that has no hand on purpose, and it connects. A stand-in would accept every width
and hold nothing while the picks report success, which is why the connect refuses it.
`Robot.from_tree(tree, gripper=None)` builds the arm alone, for a calibration.

## What a hand measures

`hold_evidence()` answers `HELD` or `EMPTY` only from a measurement, and `UNMEASURED` otherwise, and
always while the jaws are open. `width_is_measured()` says whether a width is read or only commanded.
The hand verbs read these two, never the command echoed back.

| Driver | A hold is read from | Width |
|---|---|---|
| `robotiq` | gOBJ: fingers stalled on something, or reached their target | measured |
| `onrobot` | the grip-detected bit of the status word | measured |
| `jaw_io` | the reed switches or a part sensor, where wired | commanded |
| `vacuum` | the vacuum switch, where wired | commanded |
| simulated suction | the physical bond in Isaac | commanded |
| `IsaacGripper` | nothing | measured outside mock mode |
| `dummy`, `none` | nothing | commanded |

Connecting differs on purpose. `vacuum` switches suction off at connect, because releasing a cup left
latched is cheap. A `jaw_io` hand left closed may hold a rigid part, so it opens only when feedback says
it is empty, holds and warns when feedback says a part is there, and does not move at all with no
feedback wired, unless `open_on_connect_without_feedback` is set. Both touch the controller's I/O at
connect, so the arm connects first; `Robot` does that for you.

## The simulated grippers

`IsaacGripper` drives any jaw from a `GripperProfile` (`ROBOTIQ_2F85_PROFILE`, `ROBOTIQ_HANDE_PROFILE`,
`SCHUNK_EGU50_PROFILE`, `SCHUNK_EZU35_PROFILE`), and `IsaacSuctionGripper` drives any cup from a
`SuctionCupProfile`. No vendor name reaches either, so a real cell can never build one; the Isaac
runners build them against the arm's live session, and importing `src.robot.grippers.sim` loads no Isaac.
The Hand-E has a profile and no measured mount yet, so a sim cell naming it is refused.

| Selected by | What it picks |
|---|---|
| `robot.gripper.model` | the jaw a sim cell mounts (`willy_sim.grippers.sim_mount_for`); a hand with no measured mount is refused |
| `robot.sim.suction_cup` | `STANDARD_SUCTION_CUP`, 30 mm across and the default, or `SLIM_SUCTION_CUP`, 20 mm |

A cup profile is the one source of the cup's geometry: the driver, the drawn cup and the collision
envelope read the same profile. The simulated suction bond is binary: it models the attach and the lift,
not seal quality, which is scored in [grasping/suction](../grasping/suction/README.md).

## Status

| Capability | Evidence |
|---|---|
| `IsaacGripper` and `IsaacSuctionGripper` | measured in simulation |
| The `jaw_io` and `vacuum` pins switching on a controller | measured against real controller software: URSim |
| Robotiq, OnRobot, `jaw_io` and `vacuum` on a real hand | never touched hardware |

The Robotiq arithmetic runs against a fake driver, and it is the one path URSim cannot cover, because
the URCap opens port 63352. The OnRobot driver runs against a fake Modbus client, and the I/O drivers
against a fake I/O port. None of the real drivers reports force or current.

## Files

| File | Holds |
|---|---|
| `registry.py` | `create_gripper`, `register_gripper_driver`, `available_gripper_vendors` and the rest of the registry |
| `robotiq.py`, `robotiq_socket.py` | `GripperController` and `RobotiqSocket`, the port-63352 client |
| `onrobot.py`, `onrobot_modbus.py` | `OnRobotGripper` and `OnRobotRG`, the register-level Modbus client |
| `jaw_io.py`, `vacuum.py` | `JawIOGripper` and `VacuumGripper`, over the arm's digital I/O |
| `dummy.py`, `null.py` | `DummyGripper`; `NullGripper`, `GripperSubstitution` and `SubstitutionReason` |
| `sim/` | `IsaacGripper`, `IsaacSuctionGripper` and their profiles |

## Details

- [guide 06](../../../docs/guide/06-grippers.md): choosing, wiring and driving a gripper, step by step
- [docs/runbooks/your_own_gripper.md](../../../docs/runbooks/your_own_gripper.md) and
  [hande_gripper_bringup.md](../../../docs/runbooks/hande_gripper_bringup.md)
- [robot/core](../core/README.md): the `Gripper`, `ReportsHoldEvidence`, `MeasuresWidth` and `GripperVendor` contracts
- [UR driver](../drivers/ur/README.md): the arm that owns the I/O and the bench
- [grasping/collision](../grasping/collision/README.md): the envelopes that share the cup geometry
