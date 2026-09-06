# KUKA EthernetKRL (EKI) controller-side templates

This folder contains the KRL + XML side of the Willy <-> KUKA bridge.
The matching Python driver lives at `src/robot/drivers/kuka/`.

| File | Where it goes on the controller | What it does |
|---|---|---|
| `EkiHwInterface.xml` | `KRC:\R1\TP\EthernetKRL\Willy\EkiHwInterface.xml` | Defines the EKI channel `Willy`, sets the TCP endpoint of the Python application, and routes inbound `<Cmd>` tags to flag bits. |
| `Willy.dat` | `KRC:\R1\Program\Willy.dat` | Persistent defaults (home pose, tool/base index, default vel/acc). Edit-in-place to retune without recompiling. |
| `Willy.src` | `KRC:\R1\Program\Willy.src` | KRL program implementing the wire protocol: dispatch loop, motion commands, FK/IK round-trips, telemetry publisher. |

## Wire protocol (mirror of `protocol.py`)

* Frames are newline-terminated XML.
* Willy -> KRL frames use `<Cmd Type="Willy">` as the root tag.
* KRL -> Willy frames use `<Sen Type="Willy">` as the root tag.
* Each frame **must** be followed by exactly one `\n` byte. The
  `EKI_Send` calls in `Willy.src` already append `Chr(10)`.

The complete tag list and their attributes is documented at the top of
[`src/robot/drivers/kuka/protocol.py`](../../../../src/robot/drivers/kuka/protocol.py)
and in the header of [`Willy.src`](Willy.src).

## Connection direction

Willy's default Python configuration (`KukaEkiConfig.role = "server"`)
expects the **KRL program** to dial in. That matches the
`<EXTERNAL><TYPE>Client</TYPE>` block of `EkiHwInterface.xml`. Set
`WILLY_HOST` / `WILLY_PORT` on the controller to the IP / port of
the machine running Willy.

If your safety case requires the controller to listen instead, flip
`role` in `robot.yaml` to `"client"`, set the controller's IP under
`robot.connection.ip`, and switch the EKI XML to
`<TYPE>Server</TYPE>`.

## Required controller options

* **EthernetKRL** option installed on the controller.
* For FK / IK round-trips: `$POS_FOR()` and `INVERSE()` are part of
  base KSS, no extra option is required.
* Default base / tool indices used by `Willy.src` are configured via
  `WILLY_BASE_NO` / `WILLY_TOOL_NO` in `Willy.dat`. Make sure the
  matching `BASE_DATA[]` / `TOOL_DATA[]` entries are calibrated.

## Quick sanity test

After loading the files and starting `Willy()`, point the Python side
at the same TCP endpoint and run:

```python
from src.config.schema.robot import RobotConfig
from src.robot.drivers import create_arm
from src.robot.core import RobotVendor

cfg = RobotConfig(vendor="kuka")  # or load_robot_config() from a profile
arm = create_arm(RobotVendor.KUKA, config=cfg)
arm.connect()
print(arm.get_tcp_pose())
arm.disconnect()
```

If `get_tcp_pose()` returns a `Pose` in `Frame.BASE` the protocol is
wired up correctly.
