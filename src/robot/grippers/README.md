# robot.grippers

Every end-effector, real or simulated, behind the one vendor-neutral `Gripper` Protocol.

## What this package guarantees

The concrete drivers live here and satisfy the `Gripper` Protocol defined in
[`../core/`](../core/README.md), so pipeline code never imports a vendor SDK and never branches on
which end-effector is attached. Real drivers are reached through a lazy factory registry,
`create_gripper(vendor, **kwargs)`, whose factory bodies defer any transport import until they run.
The simulator drivers live in `sim/` and are selected by config too, but deliberately not through the
registry.

Adding a gripper means adding a `GripperVendor` member and one module here. No pipeline changes.

## Contents

| Path | Role |
| --- | --- |
| `registry.py` | Maps a `GripperVendor` to a lazy factory. `create_gripper()` builds it and then checks the result against the Protocol. |
| `robotiq.py` | `GripperController`, a Robotiq 2F-85, 2F-140 or Hand-E over the URCap socket. |
| `robotiq_socket.py` | `RobotiqSocket`, a dependency-free client that speaks the port-63352 grammar directly. |
| `onrobot.py` | `OnRobotGripper`, an RG2 or RG6 over Modbus TCP through the Compute Box. |
| `onrobot_modbus.py` | `OnRobotRG`, `RGStatus` and `ModbusError`, the register-level client. |
| `vacuum.py` | `VacuumGripper`, suction over the controller's digital I/O. |
| `jaw_io.py` | `JawIOGripper`, parallel jaws over the controller's digital I/O, single or double solenoid. |
| `dummy.py` | `DummyGripper`, pure Python for offline work. It tracks a width in memory and clamps to range. |
| `null.py` | `NullGripper`, the explicit no-op, plus `GripperSubstitution` and `SubstitutionReason`. |
| `sim/gripper.py` | `IsaacGripper`, a parallel jaw driven by a swappable `GripperProfile`. |
| `sim/suction_gripper.py` | `IsaacSuctionGripper`, a simulated surface gripper driven by a `SuctionCupProfile`. |

`__init__.py` registers six built-ins: `ROBOTIQ`, `ONROBOT`, `VACUUM`, `JAW_IO`, `DUMMY` and `NONE`.

## How a gripper is chosen

| Gripper | Selected by | Mechanism |
| --- | --- | --- |
| Real hardware | `robot.gripper.vendor` | `create_gripper(vendor, **kwargs)` looks up the lazy factory and checks the result against the Protocol |
| Simulated parallel jaw | `robot.sim.gripper_mount`, for example `schunk_egu50` | picks a `MountedGripperSpec` from `willy_sim.grippers.MOUNTED_GRIPPERS` and uses its `GripperProfile`; the default is the baked Robotiq 2F-85 |
| Simulated suction cup | `robot.sim.suction_cup`, for example `slim` | picks a `SuctionCupProfile` from `willy_sim.grippers.SUCTION_CUPS`; the default is the standard cup |

The rule is the same in every row: a new gripper, or a new cup, is data in the shape of a profile,
not new driver code. One `IsaacGripper` drives any jaw profile and one `IsaacSuctionGripper` drives
any cup profile.

## The three wire protocols, and why they are separate drivers

`robotiq` speaks a socket protocol on TCP port 63352 that the URCap opens on the UR controller. There
is no SDK behind it: `robotiq_socket.py` speaks that grammar directly. Native position counts from 0
to 255 convert to millimetres inside the driver, and `speed` and `force` are normalised to the range
0 to 1 and mapped onto counts, so they are not newtons.

`onrobot` speaks Modbus TCP to an OnRobot Compute Box, which is a separate device on its own network
address rather than something the arm hosts. Three things are the opposite of the Robotiq: there is
no activation stroke, there is no speed register, and the width is an opening in tenths of a
millimetre, so a larger number means more open. Force is newtons natively. The default port is 502
and the default unit is the quick changer's. RG2 and RG6 only: the 2FG7 shares the family name and
not the register map, and the vacuum tools and the three-finger model are different devices again.

`vacuum` and `jaw_io` are the same idea applied to different hardware: on a UR controller an ejector
and a solenoid jaw are both pins. There is no SDK and nothing manufacturer-specific in either, which
is why both are named for what they are and how they speak rather than for a maker, and why every
wiring number lives in config (`gripper.vacuum.*`, `gripper.jaw_io.*`) rather than in the driver. A
cell can be configured and tested before the end-effector has been bought, and bring-up becomes a
matter of measuring numbers rather than editing code.

Measuring those numbers has its own tool. Nothing below moves the arm, and every write is gated:

```bash
python -m src.robot.drivers.ur --read                        # read every pin on the bank, change nothing
python -m src.robot.drivers.ur --watch 0 --for 15            # trip a sensor by hand and watch a pin
python -m src.robot.drivers.ur --measure 4=1 --watch 0 --yes # drive a pin and time the response
```

## Post-close verification

`ObjectDetectingGripper` is the opt-in extension that answers whether the end-effector is holding
something. Four drivers implement it and they differ in how much they know:

- `jaw_io` reads a reed pair, which distinguishes closed on nothing from closed on a part. This is
  the first real post-close evidence available to a solenoid jaw.
- `onrobot` reads the grip-detected bit out of the status word directly.
- `vacuum` reads the vacuum switch if one is wired. Without a switch there is nothing to read, so it
  reports the commanded state, which is the honest answer as far as the driver knows.
- The simulated suction gripper polls the physical bond.

`robotiq`, `dummy` and `null` do not implement it, so a gate written against it stays inert on those.

`jaw_io` also has an unusual connect rule, and it differs from the suction driver on purpose. Suction
asserts off on connect, because releasing a cup that was left latched is cheap. A jaw gripper left
closed from an interrupted run may be holding a rigid part, and opening it drops that part wherever
the arm is standing. So: feedback says empty, open; feedback says something is held, hold it and warn
so a person decides; no feedback wired, do not actuate at all, unless
`open_on_connect_without_feedback` opts into the suction behaviour.

## The substituted gripper

A `NullGripper` is also what a caller gets when a real one could not be built, and that case must be
distinguishable from a cell that genuinely has no end-effector. Five config combinations cannot
produce a real gripper, one per `SubstitutionReason` member: an unknown vendor, Robotiq asked for on
an arm that is not a UR, vacuum or a digital-I/O jaw asked for on an arm that advertises no digital
I/O, and a recognised vendor with no driver in this repository.

Each of those otherwise produces a working `NullGripper` and a log line, so the cell connects, every
pick reports success, the jaws close on nothing and lift nothing. `GripperSubstitution` carries the
reason, the requested vendor, a sentence an operator can act on, and the fix, on the object itself,
where a caller reads it without parsing logs. A `NullGripper` whose `substitution` is `None` is a
cell that has no end-effector on purpose.

## Usage

```python
from src.robot.grippers import GripperVendor, create_gripper

# Offline work and tests.
g = create_gripper(GripperVendor.DUMMY, max_width_mm=85.0)
g.connect()
g.activate()
g.set_width_mm(40.0)
print(g.get_width_mm())

# A real Robotiq on a UR controller. The gripper is daisy-chained on the tool I/O.
from src.config.schema.robot import GripperConfig
g = create_gripper(GripperVendor.ROBOTIQ, config=GripperConfig(), ip="192.168.1.10")

# A real OnRobot RG2. The host is the Compute Box, not the robot.
g = create_gripper(GripperVendor.ONROBOT, config=GripperConfig(), host="192.168.1.20")

# Simulated suction with the slim cup, built by a sim runner rather than the registry.
from src.robot.grippers.sim import SLIM_SUCTION_CUP, IsaacSuctionGripper
g = IsaacSuctionGripper(session=session, gripper_prim_path=path, profile=SLIM_SUCTION_CUP)
```

`create_gripper(vendor, **kwargs) -> Gripper` raises `ValueError` for a vendor string that is not a
`GripperVendor`, `RobotConnectionError` for a known vendor with no registered factory, and
`TypeError` if a factory returns something that does not satisfy the Protocol.
`register_gripper_driver(vendor, *, overwrite=False)` is the registering decorator, and
`unregister_gripper_driver`, `is_gripper_vendor_registered` and `available_gripper_vendors` are the
introspection surface.

Also re-exported from this package: `Gripper` and `GripperVendor` from `robot.core`;
`GripperDriverFactory`; `GripperSubstitution` and `SubstitutionReason`; `RobotiqSocket` and
`RobotiqSocketError`; `OnRobotRG`, `RGStatus` and `ModbusError`; and `GripperController` under its own
name for existing imports.

From `src.robot.grippers.sim`: `IsaacGripper`, `GripperProfile` and the profiles
`ROBOTIQ_2F85_PROFILE`, `SCHUNK_EGU50_PROFILE` and `SCHUNK_EZU35_PROFILE`; `IsaacSuctionGripper`,
`SuctionCupProfile`, `STANDARD_SUCTION_CUP` and `SLIM_SUCTION_CUP`.

## The simulator drivers, and why a real cell cannot reach them

They are kept out of the vendor registry: no `GripperVendor` member points at them, so
`create_gripper` can never return one for a real robot. The sim runners construct them by hand with a
live simulator session, and every simulator import is deferred, so this package imports cleanly on a
machine with no simulator installed.

`IsaacGripper` drives a gripper articulation through a `GripperProfile`, which names the driven joint
and carries a measured table from joint angle to jaw width. `IsaacSuctionGripper` drives a binary
vacuum joint by reinterpreting `set_width_mm`: at or below the profile's vacuum-on threshold the
vacuum engages.

| Cup profile | Contact diameter | Use |
| --- | --- | --- |
| `STANDARD_SUCTION_CUP` | 30 mm, from a 15 mm radius | the default, matching the collision-envelope defaults |
| `SLIM_SUCTION_CUP` | 20 mm, from a 10 mm radius | tight gaps and small flat faces |

A cup profile carries its geometry, meaning cup radius, height and shaft, as the single source of
truth. The driver, the visible-cup author in the simulator layer and the collision envelope in
`grasping.collision` all read the same profile, so which cup is in use is one decision rather than
three. A profile may name a cup mesh asset for a realistic render; without one, primitive cylinders
are drawn.

## Traps

Only the registry seam, `DummyGripper`, `NullGripper` and the two simulator drivers are exercised
outside real hardware.

The Robotiq driver has never driven a physical gripper. Its millimetre-to-count arithmetic and its
injection seam are exercised with a fake driver. It is also the one path that simulated UR controller
software cannot cover, because port 63352 is opened by the URCap rather than by the robot interface,
so it stays a bench item.

The OnRobot driver has never driven a physical gripper either. It is exercised against a fake Modbus
client.

`jaw_io` and `vacuum` have never touched a real end-effector. Their logic is exercised against a fake
I/O port, and the controller pins have been observed switching against simulated controller software,
but the wiring itself, meaning which bank, active high or low, and travel time, is exactly what the
exerciser above exists to measure.

`GripperVendor` lists `FRANKA_HAND` and `SCHUNK`, and neither has a real-hardware driver.
`create_gripper(GripperVendor.FRANKA_HAND)` raises `RobotConnectionError`. Schunk is realised in
simulation only, through `robot.sim.gripper_mount: schunk_egu50` or `schunk_ezu35` on the
vendor-neutral `IsaacGripper`.

Simulated suction has a binary bond. The simulator models the attach and the lift, meaning whether
the cup holds the part through the move. It does not model seal quality. The analytical seal and
wrench score lives in [`../grasping/suction/`](../grasping/suction/README.md).

None of the real drivers reports force or current.

## See also

- [`../core/`](../core/README.md) for the `Gripper`, `ObjectDetectingGripper` and `GripperVendor` contracts
- [`../drivers/ur/`](../drivers/ur/README.md) for the arm that owns the digital I/O these drivers switch
- [`../grasping/suction/`](../grasping/suction/README.md) for the analytical suction seal and wrench score
- [`../grasping/collision/`](../grasping/collision/README.md) for the envelope models that share the cup geometry
- `docs/guide/06-grippers.md` for choosing and configuring one from scratch
