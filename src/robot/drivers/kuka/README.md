# KUKA driver (`src/robot/drivers/kuka`)

The driver for KUKA arms on a KRC4 or KRC5 controller. It speaks EthernetKRL (EKI), XML over one TCP
socket, to a small KRL program you load on the controller, and needs no KUKA software on this side.
KRL's units (E6POS, degrees, ABC Euler) stay inside this package; everything above it sees
millimetres, XYZW quaternions and base-frame poses.

> [!WARNING]
> This driver has never commanded a KUKA controller. The transport, the conversions and the guards are
> unit-tested; the wire framing, the telemetry cadence and every motion are unproven. Validate the
> controller side with an integrator before anything moves.

```python
from willy import Robot, load_tree

robot = Robot.from_tree(load_tree(), gripper=None)   # a cell with robot.vendor: kuka
with robot.connected():                  # Willy listens; the KRL program dials in
    print(robot.arm.get_tcp_pose())      # millimetres, base frame, from the controller's <State>
```

The controller half, and where each file goes on the KRC, is
[config/robot/templates/kuka/](../../../../config/robot/templates/kuka/README.md).
`create_arm("kuka", config=tree.robot)` builds the arm alone.

## The nouns

| Noun | Built by | Verb | Returns |
|---|---|---|---|
| `KukaRobotArm` | `Robot.from_tree(tree)`, or `create_arm("kuka", config=tree.robot)` | `move(pose)`, `move_to_joints(joints)`, `get_tcp_pose()` | `MotionResult`; a base-frame `Pose` |
| `EkiClient` | the arm, from `robot.kuka.eki` | `send_move_cartesian`, `send_movej`, `request_fk`, `request_ik` | the transport; runs over `socket.socketpair` in tests |
| `KukaCartesian` | `pose_to_kuka_cartesian(pose)` | `kuka_cartesian_to_pose(cartesian)` | E6POS in millimetres and ABC degrees |

The rotation convention is `R = Rz(A) * Ry(B) * Rx(C)`, intrinsic ZYX, in degrees.
`joints_rad_to_deg` and `joints_deg_to_rad` convert the joint vectors. `KUKA_CAPABILITIES` describes a
6-axis `kr6-r900`; any other `robot.kuka.model` and `dof` build their own descriptor from those two
values.

## Configuration

```yaml
robot:
  vendor: "kuka"
  workspace_limits: { ... }        # the box every Cartesian move is checked against
  safety:
    joint_limits:                  # required: no KUKA model carries a built-in table
      min_deg: [ ... ]             # one entry per axis, in degrees
      max_deg: [ ... ]
  kuka:
    model: "kr6-r900"
    dof: 6
    controller_ip: "192.168.1.20"  # read only when eki.role is "client"
    eki:
      role: "server"               # Willy listens and the KRL program dials in
      host: "0.0.0.0"
      port: 7000
      timeout_s: 5.0
      heartbeat_s: 0.5
      buffer_size: 65536
```

The schema is `KukaConfig` in [kuka_schema.py](../../../config/schema/robot/kuka_schema.py); its
default `model` is `"unconfigured"`, so name yours. `config/robot/templates/kuka_eki.yaml` is a longer
starting point; the schema refuses its `controller` and `transport` lines, so leave those two out.

## What it refuses

| Refusal | When | What to do |
|---|---|---|
| `JointLimitTableMissing` | at build, until `robot.safety.joint_limits.min_deg` and `max_deg` are both set | Set both from your arm's software limits |
| a substituted gripper, then `NoRealGripper` at connect | the cell names `robotiq`, `jaw_io` or `vacuum`: this arm is not a UR and offers no digital I/O | Use `onrobot` or `none`, or build with `gripper=None` |
| a preflight `MotionStatus`, such as `WORKSPACE_REJECTED` | a pose the `SafetyPreflight` refuses, checked before a frame is sent | Read the status; move the target or correct the box |
| `INVALID_TARGET`, `CONNECTION_ERROR` | a pose not in `Frame.BASE`, or a move while the link is down | Tag poses with the base frame; reconnect |
| `RobotConnectionError` | a state read before the link is open, or before the first `<State>` frame arrived | Check that the KRL program runs and dials the right address |

`move()` returns these as a `MotionResult`; `move_linear` and `move_joint` raise `RobotMotionRejected`.
When the link dies, every pending request fails at once instead of waiting out its timeout. No
`<EchoAck>` within twice `heartbeat_s`, or `timeout_s` if that is longer, counts as a dead link.

## Wire protocol

Newline-terminated XML frames. Commands to the controller have the root `<Cmd Type="Willy">`,
telemetry and replies from KRL the root `<Sen Type="Willy">`.

| Direction | Tag | Purpose |
|---|---|---|
| to KRL | `<Move Mode="PTP\|LIN" X..C Vel Acc/>` | Cartesian move |
| to KRL | `<MoveJ J1..J6 Vel Acc/>` | joint move, in degrees |
| to KRL | `<Stop/>`, `<Home/>` | abort, or go to the controller's home |
| to KRL | `<FkRequest Id J1..J6/>`, `<IkRequest Id X..C S1..S6/>` | FK and IK on the controller |
| to KRL | `<Echo Token/>` | heartbeat |
| from KRL | `<State X..C J1..J6 Steady="0\|1"/>` | telemetry, continuous |
| from KRL | `<Ack Cmd Status Detail/>` | command result |
| from KRL | `<FkResult .../>`, `<IkResult .../>`, `<EchoAck Token/>` | replies |

## Status

| Capability | Evidence |
|---|---|
| The `Pose` against E6POS and joint conversions | never touched hardware: pinned by unit tests |
| The fail-fast path when the peer closes the socket | never touched hardware: pinned against a fake peer |
| The EKI framing, the `<State>` cadence, FK and IK on a real KRC, every motion | never touched hardware |
| The KRL program and the EKI channel under `config/robot/templates/kuka/` | never touched hardware |

What the driver does not do: it writes no payload to the controller (whatever WorkVisual holds is what
the controller uses), it ships no KUKA self-collision model, and its capability flags state intent:
`has_native_fk` and `has_native_ik` name KRL round trips that have not run, and there is no
asynchronous move and no force control.

## Files

| File | Holds |
|---|---|
| `arm.py` | `KukaRobotArm` and `KUKA_CAPABILITIES`: the transport, the workspace guard and the preflight together |
| `eki_client.py` | `EkiClient`: the reader thread, the telemetry cache, the request and heartbeat bookkeeping |
| `protocol.py` | the XML encode and decode, with no I/O |
| `pose_convert.py` | `KukaCartesian` and the pose and joint conversions |

## Details

- [drivers](../README.md): the registry, the doctor and the other vendors
- [robot/core](../../core/README.md): the `RobotArm` Protocol and `MotionResult`
- [safety](../../safety/README.md): `WorkspaceGuard` and the `SafetyPreflight` every motion passes
- [docs/runbooks/cell_bringup.md](../../../../docs/runbooks/cell_bringup.md): bringing a cell up
