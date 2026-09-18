# KUKA controller templates (`config/robot/templates/kuka`)

The controller half of the KUKA driver: a KRL program and an EthernetKRL channel definition you copy
onto a KRC4 or KRC5, so the Python driver in [src/robot/drivers/kuka](../../../../src/robot/drivers/kuka/README.md)
has something to talk to. These files have never run on a controller; an integrator validates them
before anything moves.

| File | Where it goes on the controller | What it does |
|---|---|---|
| [`EkiHwInterface.xml`](EkiHwInterface.xml) | `KRC:\R1\TP\EthernetKRL\Willy\EkiHwInterface.xml` | defines the EKI channel `Willy`, the TCP endpoint of the Python side, and the inbound tags |
| [`Willy.dat`](Willy.dat) | `KRC:\R1\Program\Willy.dat` | defaults: home pose, base and tool index, velocity and acceleration; edit in place |
| [`Willy.src`](Willy.src) | `KRC:\R1\Program\Willy.src` | the program: dispatch loop, motion commands, FK and IK round trips, telemetry |

## Before you load them

- The controller needs the EthernetKRL option. The FK and IK round trips use `$POS_FOR()` and
  `INVERSE()` from the base system software.
- `Willy.dat` sets the base and tool the program uses, `WILLY_BASE_NO` and `WILLY_TOOL_NO` (both 1).
  Calibrate the matching `BASE_DATA[]` and `TOOL_DATA[]` entries first.
- Replace `${WILLY_HOST}` and `${WILLY_PORT}` in `EkiHwInterface.xml` with the address of the machine
  running Willy and the port it listens on, `robot.kuka.eki.port` (7000 by default).

## Which side dials

By default Willy listens and the KRL program dials in: `robot.kuka.eki.role: "server"` on the Python
side, `<TYPE>Client</TYPE>` in `EkiHwInterface.xml`. If your safety case needs the controller to
listen instead, set `robot.kuka.eki.role: "client"`, put the controller's address in
`robot.kuka.controller_ip`, and change the XML to `<TYPE>Server</TYPE>`.

## The wire

Every frame is XML followed by exactly one newline byte; the `EKI_Send` calls in `Willy.src` append
`Chr(10)`. Frames to the controller have the root `<Cmd Type="Willy">`, frames from it
`<Sen Type="Willy">`. The tags and their attributes are listed at the top of
[`protocol.py`](../../../../src/robot/drivers/kuka/protocol.py) and in the header of
[`Willy.src`](Willy.src), and summarised in the [driver README](../../../../src/robot/drivers/kuka/README.md).

## The first check

With the files loaded and `Willy()` running, load your cell's profile on the Python side and read the
pose back. Nothing moves:

```python
from willy import Robot, load_tree

robot = Robot.from_tree(load_tree(), gripper=None)   # your KUKA cell's profile
with robot.connected():
    print(robot.arm.get_tcp_pose())
```

A `Pose` in the base frame means the channel is wired. A `JointLimitTableMissing` on the first line
means the profile does not set `robot.safety.joint_limits.min_deg` and `max_deg` yet; no KUKA model
carries a built-in table. [kuka_eki.yaml](../kuka_eki.yaml) next to this folder is a starting point for
the profile: it sets no joint limits, and the schema refuses its `controller` and `transport` lines, so
leave those two out.
