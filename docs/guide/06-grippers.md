# 6. Driving a gripper

[04](04-robot-and-safety.md) covers how to *declare* a gripper. This one is about making one actually
move: which of the four real-hardware drivers yours is, what you must measure before commanding
anything, and how to drive it from a terminal and from Python.

Nothing on this page has gripped anything. The wire formats are pinned against reference clients and
simulated controller software; the wiring, the register meanings on a live tool, and the end-effector
itself are the measurements you make. Every section says which is which.

> **The first command you send a gripper should never be a closing one.** Two of the four drivers have
> opposite polarity from each other, and one vendor's own reference client contradicts itself about
> which end is open. Every family below has a read-only first step. Use it.

Sibling guides: [01](01-configuration.md) . [02](02-models.md) . [03](03-calibration.md) .
[04](04-robot-and-safety.md) . [05](05-pick-loop.md) . **06**

---

## 1. Which driver is yours

| Your gripper | `robot.gripper.vendor` | Speaks | Needs |
|---|---|---|---|
| Robotiq 2F-85, 2F-140, Hand-E | `robotiq` | ASCII over TCP 63352 on the **UR controller** | the Robotiq URCap installed |
| OnRobot RG2, RG6 | `onrobot` | Modbus TCP to the **Compute Box** | the box on your network |
| Any two-state gripper on the tool I/O (pneumatic, electric, most makes) | `jaw_io` | one or two output pins | wiring only |
| A suction cup and ejector | `vacuum` | one output pin, optionally a vacuum switch | wiring only |
| Nothing attached | `none` | | |

`dummy` is the sixth registered vendor, for offline work. `franka_hand` and `schunk` are declared
enum names with no driver: `create_gripper` on either raises `RobotConnectionError`. A Schunk on the
tool I/O works today through `jaw_io`.

**None of the four real drivers needs a third-party package.**
[`src/robot/grippers/robotiq_socket.py`](../../src/robot/grippers/robotiq_socket.py) speaks the port-63352
grammar directly, [`onrobot_modbus.py`](../../src/robot/grippers/onrobot_modbus.py) speaks Modbus
directly, and the two digital-I/O drivers switch pins through the arm. So on a bare checkout
`python -m src.robot.drivers.doctor` reports every gripper row as ready; the rows that are not ready
are arm rows whose SDK is missing, and the two reserved gripper slots.

> **`jaw_io` is the one most people need and the one nobody looks for.** It is deliberately not named
> after a manufacturer: a pneumatic or electric two-finger gripper on a UR is one or two output pins
> plus, usually, reed switches on inputs. If your gripper is not a Robotiq or an OnRobot, try this
> before assuming you need a new driver.

---

## 2. The ladder, and why it is a ladder

Each rung answers one question and cannot answer the next. Skipping a rung means the first thing you
learn is learned with a powered gripper in the room.

```
1. does anything answer?            read-only, no motion
2. what does it say after it is
   moved by hand or from the pendant?  read-only, no motion  <- polarity settled here
3. one small opening command        first motion, hand clear
4. a close, on nothing              first grip
5. a close, on a part               the real thing
```

Rungs 1 and 2 are read-only by construction, not by convention: none of the probe entry points has a
flag that makes them write.

---

## 3. Robotiq (2F-85, 2F-140, Hand-E)

The wire format is pinned against a reference client's grammar and the client runs against a TCP
server implementing that grammar. Nothing here has spoken to a real URCap, and simulated UR controller
software cannot cover it either, because port 63352 is opened by the URCap rather than by the robot
interface.

### Rung 1: does anything answer

```bash
python scripts/ursim/probe_robotiq_urcap.py 192.168.1.10
```

> **A successful TCP connect proves nothing here, and the probe says so out loud.** Against a
> controller with port 63352 published and no URCap installed, the connect succeeds and the connection
> dies on the first byte: a port scanner says yes and there is no gripper. Only a parseable reply is
> evidence.

If it refuses, the three things to check are named in the message: the URCap installed, Tool I/O set
to the Robotiq grippers, and the RS-485 URCap **not** installed, since the two are mutually exclusive.

### Rung 2: settle the polarity yourself

Run the probe and note `POS`. Move the fingers from the teach pendant. Run it again.

**`POS` must go up as the fingers close.** The vendor documents zero as open for the same register
repeatedly, and a widely used third-party header declares its device-unit zero as open while its
normalised zero is closed. Anyone lifting a snippet from that project without checking which unit mode
it is in inverts the jaws. This client is device units only.

### What activation does

`connect()` runs a calibration routine that **sweeps the fingers through their entire travel, at
speed.** Nothing may be between the jaws. This is why the operator console refuses a connect without
an acknowledged preview that named that specific motion.

### Two traps in the protocol itself

| | |
|---|---|
| **`POS` is two registers** | Written it is the *target*; read it is the *actual position*. Setting a position and then reading it back does not return what was written. The echo of the request is a different variable, `PRE`. |
| **Speed `0` is minimum speed, not stop** | The register maps 0 to 100 percent onto a non-zero band. To hold the fingers still, clear `GTO`. |

### Millimetres

The driver maps millimetres to counts from the configured `max_width_mm`. That map is approximate and
the vendor's own numbers do not close: a 2F-85 manual states 0.4 mm per count over an 85 mm stroke,
and 255 counts at 0.4 mm is 102 mm. The real endpoints are a per-unit measurement, which is why other
projects ship an auto-calibration for exactly this. Treat a commanded millimetre as approximate until
you have measured your unit.

---

## 4. OnRobot (RG2, RG6)

The Modbus framing was measured against a real Modbus TCP server. The *register meanings* reproduce
from the vendor manual, were re-derived adversarially, and agree with two independent open-source
drivers, but agreement is not measurement, and no Compute Box has been in front of this code.

**RG2 and RG6 only.** The 2FG7 shares the family name and not the register map, and no public map for
it could be sourced. The vacuum tools are a different device and the three-finger model is another
again. A driver that accepted them would be guessing at addresses, and a guessed Modbus address either
reads a plausible number or writes into something else, and both look like success.

```yaml
robot:
  gripper:
    vendor: onrobot
    max_width_mm: 160.0      # RG6; an RG2 is 110
    min_width_mm: 0.0
    onrobot:
      host: 192.168.1.1      # the Compute Box, not the robot
      unit_id: 65            # chosen by the mounting, not by the tool
      default_force_n: 20.0  # newtons, natively
```

### Rung 1: read, and sweep the unit ids

```python
from src.robot.grippers.onrobot_modbus import OnRobotRG

rg = OnRobotRG()
rg.connect("192.168.1.1")
print(rg.probe().render())     # reads only; also tries units 65, 66 and 67
rg.disconnect()
```

The sweep costs three frames, moves nothing, and settles empirically which side of a Dual Quick
Changer your tool is on, instead of by reading a table.

> **A force-torque variant answers on the same unit id with an incompatible map.** The id is chosen by
> the mounting, not by the tool, so such a variant in the same Quick Changer replies to every read
> looking perfectly healthy while meaning something else, a force value landing in a width field.
> There is no way to detect this from Modbus alone. Confirm the model in the Compute Box web client
> before commanding anything.

### Rung 2: polarity and units, with calipers

Read register 267 with the fingers parked mid-stroke and measure the jaw gap. The register must read
roughly the gap in tenths of a millimetre. Then command a small *opening* move and confirm the jaws
move outward.

**The polarity is the opposite of the Robotiq's.** Here the number is an *opening* in tenths of a
millimetre, so bigger is more open. Porting a helper between the two drivers inverts a gripper.

### Three things an RG does not have

| | |
|---|---|
| **No activation** | `connect()` reads two registers and commands nothing. `activate()` is a deliberate no-op. |
| **No speed register** | `set_width_mm(speed=...)` accepts the argument, drops it, and logs that once per connection. Raising would break every vendor-neutral caller on a parameter the Protocol requires. |
| **No address of its own** | The gripper has none; the Compute Box does. Never point `host` at the arm. |

A write echo means accepted, not moving and not finished: the gripper ignores a grip command while the
busy flag is set, and an ignored grip still returns a normal echo. Poll the status word. And a tripped
safety switch leaves the gripper dead until tool power is cycled, with every command meanwhile
silently discarded, which is why `connect()` refuses on it: a message now, or a dropped part later.

---

## 5. Digital-I/O grippers (`jaw_io`, `vacuum`)

The logic is exercised against a fake I/O port, and the controller pins have been observed switching
against simulated controller software. The wiring is what the bench tool exists to measure.

This is the family where the numbers are yours: which pin, which bank, active high or low, how long
the cylinder takes. All of it is config, on purpose, so the drivers could be written before the
hardware was chosen. That trade only pays off if the numbers are cheap to obtain.

### The bench

```bash
python -m src.robot.drivers.ur --read                        # read every pin, change nothing
python -m src.robot.drivers.ur --watch 0 --for 15            # trip a sensor by hand, see it
python -m src.robot.drivers.ur --measure 4=1 --watch 0 --yes # the number you came for
```

`--measure` drives an output, times how long the watched input takes to answer, and prints the
milliseconds that become `close_settle_s` and `engage_timeout_s`. It never moves the arm.

> **Run it a few times and configure the worst reading.** The driver waits that budget out, so a
> typical value turns into a dropped part on a slow stroke.

Every write is gated behind `--yes`, and the gate refuses rather than prompts in a non-interactive
shell: a prompt that reads end-of-file and takes the default is how an unattended script drives a coil
nobody authorised. From Python the gate is a required `confirm` argument with no default, section 6.

`--port` is not optional in spirit. Standard output 4 is in the control cabinet; tool output 4 is on
the wrist. A tool-mounted gripper is usually on `tool`, which is what both config blocks default to.

### What `connect()` does, and it depends on your wiring

`jaw_io` is the one gripper whose connect behaviour is decided by what you wired:

| Your cell | What `connect()` does |
|---|---|
| end-stop switches wired | reads them; **opens only on an empty reading**, holds a detected part and warns so a person decides |
| no switches, `open_on_connect_without_feedback: true` | **opens unconditionally**, and anything held is dropped |
| no switches, flag off | does not actuate at all |

`vacuum` differs on purpose: it asserts off on connect, because releasing a latched cup is cheap,
while opening jaws that hold a rigid part drops it wherever the arm is standing. Its `disconnect()`
never raises, because a teardown path must not strand a held part.

### The feedback is the real payoff

With a vacuum switch wired, `is_object_detected()` is a measurement rather than the commanded state.
The jaw driver answers the same question from whatever is wired, in a fixed order: a dedicated
part-present pin wins outright; failing that, a fully-open **and** a fully-closed reed switch let it
infer the answer, since "fully closed" means the jaws met each other and the grasp is empty while
"stopped in between" means something is between them; failing both, it returns the **commanded**
state. Wire nothing and you get no measurement, only a politer restatement of what you asked for.

---

## 6. From Python

Every gripper satisfies the same vendor-neutral `Gripper` Protocol, so a caller that holds one does
not know which of the six it has:

```python
from src.config import load_robot_config
from src.robot.execution.cell import Cell

cell = Cell.from_robot_config(load_robot_config())
cell.build()
with cell.connected() as live:
    gripper = live.service.runtime.orchestrator.gripper
    gripper.set_width_mm(40.0)
```

The bench is a noun too, and its interlock is not optional:

```python
from src.robot.drivers.ur.bench import Bench, Measure

bench = Bench.from_robot_config(robot_cfg, arm, confirm=lambda what: input(f"{what}? ") == "yes")
print(bench.run(Measure(pin=4, level=True, watching=0)).render())
```

`confirm` has no default. The functions underneath have no gate at all, since they energise a pin the
moment they are called, so a default that returned `True` would be the absence of an interlock wearing
its name. `Bench.from_robot_config` also refuses to guess the bank: a config whose gripper is neither
`jaw_io` nor `vacuum` declares no bank, and the honest answer is that this cell has no digital-I/O
gripper to exercise.

And one action at a time. `Bench.run` takes a single action object, `Read`, `Watch`, `Set`, `Pulse` or
`Measure`, so "read and also set" cannot be expressed, and the command line refuses the same
combination rather than silently honouring one flag and dropping the other.

---

## 7. When it does not work

| Symptom | Most likely |
|---|---|
| Everything connects, every pick reports success, nothing is ever held | A **substituted `NullGripper`**. Read `gripper.substitution` or the build warning. This is the single most dangerous silent state in the stack. |
| Robotiq: connect works, every read times out | The URCap daemon is up and its RS-485 link to the gripper is dead. |
| Robotiq: the connection dies on the first byte | Nothing is listening behind a forwarded port: the URCap is not installed or not running. |
| OnRobot: silence | Silence is the documented symptom of *every* misconfiguration: wrong unit id, Modbus not served by this firmware, or no tool on that Quick Changer side. |
| Digital I/O: the write is accepted and the pin does not move | Wrong bank. Try `--port standard`, `configurable` or `tool`. |
| Digital I/O: the output was driven and the input never answered | Wrong pin, unpowered sensor, or an active-low switch. |

> **The first symptom deserves its own paragraph.** `from_robot_config` answers a misconfigured
> end-effector with a working `NullGripper` rather than crashing, so the cell comes up, every pick
> reports success, and the jaws close on nothing. Five config combinations trigger it, one per
> substitution reason: an unrecognised vendor name; `robotiq` on an arm that is not a UR; `vacuum` on
> an arm with no digital I/O; `jaw_io` on an arm with no digital I/O; and a recognised name with no
> driver. `onrobot` has no such precondition, because the Compute Box does not depend on the arm at
> all. The substitution object carries the reason, what was requested, a sentence an operator can act
> on, and the fix. A `NullGripper` whose `substitution` is `None` is a cell that has no end-effector on
> purpose.

---

## 8. What is not here

- **No three-finger gripper, no vacuum URCap tools, no 2FG7.** Each is a different protocol or an
  unsourced register map. A vacuum URCap is the dangerous near-miss: it installs a *different* URCap
  and reuses the same variable names with different meanings, where `POS` is a vacuum level.
- **No Schunk or Franka driver.** Both are declared enum names with no implementation. A Schunk on the
  tool I/O works today through `jaw_io`; in simulation it is `robot.sim.gripper_mount: schunk_egu50`
  or `schunk_ezu35`, driven by the vendor-neutral simulated jaw gripper through a profile.
- **No force or current reading from any real driver.**
- **Nothing has gripped anything.** The bench, the probes and the read-only rungs exist because the
  measurement is yours to make.

## Where to look next

- [`src/robot/grippers/README.md`](../../src/robot/grippers/README.md), every driver, the substitution rules and
  the simulator profiles
- [`src/robot/core/README.md`](../../src/robot/core/README.md), the `Gripper` and `ObjectDetectingGripper`
  contracts
- [`src/robot/drivers/ur/README.md`](../../src/robot/drivers/ur/README.md), the arm that owns the digital I/O
  these drivers switch
- [`scripts/ursim/README.md`](../../scripts/ursim/README.md), the probes in the order they are meant
  to be run
- [04](04-robot-and-safety.md) for declaring one, [05](05-pick-loop.md) for the pick that closes it
