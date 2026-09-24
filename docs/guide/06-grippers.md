# 6. Driving a gripper

[04](04-robot-and-safety.md) covers how to *declare* a gripper. This guide makes one move: which of the
real-hardware drivers yours is, what to measure before you command anything, and how to drive it from a
terminal and from Python.

```python
from willy import Robot, load_tree

robot = Robot.from_tree(load_tree())   # the cell WILLY_PROFILE names, which names its hand
with robot.connected():                # the lock, the arm, then the hand
    print(robot.grasp(40.0))           # close to 40 mm, and what the hand measured
    print(robot.is_holding())          # reads the hand again, commands nothing
    print(robot.release())             # open to the hand's width
```

That is [`examples/real_robot/04_open_and_close_the_hand.py`](../../examples/real_robot/04_open_and_close_the_hand.py);
at a desk, `load_tree("console_dummy")` runs it on a dummy hand.

The gripper drivers never touched hardware. The wire formats are pinned against reference clients, and the
OnRobot framing and the digital-I/O pins are measured against real controller software. The wiring, the
register meanings on a live tool and the end-effector itself are measurements you make. Every section says
which is which.

> [!WARNING]
> **The first command you send a gripper should never be a closing one.** Two of the drivers have
> opposite polarity from each other, and one vendor's own reference client contradicts itself about which
> end is open. Every family below has a read-only first step. Use it.

---

## 1. Which driver is yours

| Your gripper | `robot.gripper.vendor` | Speaks | Needs |
|---|---|---|---|
| Robotiq 2F-85, 2F-140, Hand-E | `robotiq` | ASCII over TCP 63352 on the **UR controller** | the Robotiq URCap installed |
| OnRobot RG2, RG6 | `onrobot` | Modbus TCP to the **Compute Box** | the box on your network |
| Any two-state gripper on the tool I/O (pneumatic, electric, most makes) | `jaw_io` | one or two output pins | wiring only |
| A suction cup and ejector | `vacuum` | one output pin, optionally a vacuum switch | wiring only |
| Nothing attached | `none` | | |

`dummy` is registered too, for offline work. `franka_hand` and `schunk` are declared names with no driver:
`create_gripper` on either raises `RobotConnectionError`. A Schunk on the tool I/O works through `jaw_io`.

**None of the real drivers needs a third-party package.**
[`src/robot/grippers/robotiq_socket.py`](../../src/robot/grippers/robotiq_socket.py) speaks the port-63352
grammar directly, [`onrobot_modbus.py`](../../src/robot/grippers/onrobot_modbus.py) speaks Modbus
directly, and the two digital-I/O drivers switch pins through the arm. So on a bare checkout
`python -m src.robot.drivers.doctor` reports every gripper row with a driver as ready; the rows that are
not ready are arm rows whose SDK is missing, and the two reserved gripper names.

**`robot.gripper.vendor` picks the driver; `robot.gripper.model` names the hand.** The hand's geometry lives
in the gripper registry, one file per hand under [`config/grippers/`](../../config/grippers/), read by
[`src/config/grippers.py`](../../src/config/grippers.py). The repository ships `robotiq_2f85`,
`robotiq_hande` and `schunk_egu50`. `robot.gripper.model` is the one name the exact-mesh guard and the
planner take the hand from: the planner's descriptor is per arm with no hand, and the hand is added as a
body link when the planner starts. The loader fills the hand's widths and collision envelope from its file,
and `python -m src.config` refuses a registry file that does not validate.

The two names meet at the desk. `--check` states the driver the build constructs in a `gripper driver` row,
and blocks exactly where the build would put a `NullGripper` on the flange, because both read one function
(`robot_parts.gripper_driver_verdict`). A registry file may list the drivers that can actuate its hand, as
`drivers: [robotiq]` does for both Robotiq hands, and a real UR profile that names such a hand with another
`gripper.vendor` is refused at load. `none` and `dummy` always pass. A file that lists nothing, like the
EGU-50's, refuses no driver.

A parallel jaw this repository does not ship joins the registry through
[your_own_gripper.md](../runbooks/your_own_gripper.md): a registry file, a body from its numbers, from
vendor STL or OBJ files or from a USD, a sphere map, a retract, an evidence file and a cell layer, each
written by one script. That chain never touched hardware: it is exercised in a copy of the tree, up to a
planner that starts.

**`jaw_io` is the one most people need and the one nobody looks for.** It is not named after a
manufacturer: a pneumatic or electric two-finger gripper on a UR is one or two output pins plus, usually,
reed switches on inputs. If your gripper is not a Robotiq or an OnRobot, try this before assuming you need a
new driver.

---

## 2. The ladder, and why it is a ladder

Each rung answers one question and cannot answer the next. Skip a rung, and the first thing you learn is
learned with a powered gripper in the room.

```
1. does anything answer?                read-only, no motion
2. what does it say after it is
   moved by hand or from the pendant?   read-only, no motion  <- polarity settled here
3. one small opening command            first motion, hand clear
4. a close, on nothing                  first grip
5. a close, on a part                   the real thing
```

Rungs 1 and 2 are read-only by construction: none of the probe entry points has a flag that makes it write.

---

## 3. Robotiq (2F-85, 2F-140, Hand-E)

Never touched hardware. The wire format is pinned against a reference client's grammar, and the client runs
against a TCP server that implements that grammar. No real URCap has answered it, and URSim cannot stand in,
because port 63352 is opened by the URCap rather than by the robot interface.

### Rung 1: does anything answer

```bash
python scripts/ursim/probe_robotiq_urcap.py 192.168.1.10
```

A successful TCP connect proves nothing here, and the probe says so. Against a controller with port 63352
published and no URCap installed, the connect succeeds and the connection dies on the first byte: a port
scanner says yes and there is no gripper. Only a parseable reply counts.

If it refuses, the message names three things to check: the URCap installed, Tool I/O set to the Robotiq
grippers, and the RS-485 URCap **not** installed, because the two exclude each other.

### Rung 2: settle the polarity yourself

Run the probe and note `POS`. Move the fingers from the teach pendant. Run it again.

**`POS` must go up as the fingers close.** The vendor documents zero as open for this register, and a widely
used third-party header declares its device-unit zero as open while its normalised zero is closed. A
snippet lifted from that project without checking its unit mode inverts the jaws. This client uses device
units only.

### What activation does

`connect()` runs a calibration routine that **sweeps the fingers through their entire travel, at speed.**
Nothing may be between the jaws. This is why the operator console refuses a connect without an
acknowledged preview that named that motion.

### Two traps in the protocol itself

| Trap | What it means |
|---|---|
| **`POS` is two registers** | Written, it is the *target*; read, it is the *actual position*. The echo of the request is `PRE`. |
| **Speed `0` is minimum speed, not stop** | The register maps 0 to 100 percent onto a band above zero. To hold the fingers still, clear `GTO`. |

### Millimetres

The driver maps millimetres to counts from the configured `max_width_mm`. That map is approximate, and the
vendor's own numbers do not close: a 2F-85 manual states 0.4 mm per count over an 85 mm stroke, and 255
counts at 0.4 mm is 102 mm. The real endpoints are a per-unit measurement, which is why other projects ship
an auto-calibration for exactly this. Treat a commanded millimetre as approximate until you have measured
your unit.

---

## 4. OnRobot (RG2, RG6)

The Modbus framing is measured against real controller software: a real Modbus TCP server. The
*register meanings* follow the vendor manual, were re-derived independently and
agree with two independent open-source drivers, but agreement is not measurement: no Compute Box has been
in front of this code, so the register meanings never touched hardware.

**RG2 and RG6 only.** The 2FG7 shares the family name and not the register map, and no public map for it
could be sourced. The vacuum tools are a different device, and the three-finger model is another again. A
driver that accepted them would be guessing at addresses, and a guessed Modbus address either reads a
plausible number or writes into something else. Both look like success.

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

The sweep costs three frames, moves nothing, and settles by measurement which side of a Dual Quick Changer
your tool is on.

A force-torque variant answers on the same unit id with an incompatible map. The id is chosen by the
mounting, not by the tool, so such a variant in the same Quick Changer replies to every read looking healthy
while meaning something else, such as a force value in a width field. Modbus alone cannot detect this.
Confirm the model in the Compute Box web client before you command anything.

### Rung 2: polarity and units, with calipers

Read register 267 with the fingers parked mid-stroke and measure the jaw gap. The register must read roughly
the gap in tenths of a millimetre. Then command a small *opening* move and confirm the jaws move outward.

**The polarity is the opposite of the Robotiq's.** Here the number is an *opening* in tenths of a
millimetre, so bigger is more open. A helper ported between the two drivers inverts a gripper.

### Three things an RG does not have

| Missing | What it means |
|---|---|
| **No activation** | `connect()` reads two registers and commands nothing. `activate()` does nothing on purpose. |
| **No speed register** | `set_width_mm(speed=...)` accepts the argument, drops it, and logs that once per connection. |
| **No address of its own** | The gripper has none; the Compute Box does. Never point `host` at the arm. |

The speed argument is accepted rather than refused because the `Gripper` Protocol requires it, and a raise
would break every vendor-neutral caller.

A write echo means accepted, not moving and not finished: the gripper ignores a grip command while its busy
flag is set, and an ignored grip still returns a normal echo, so poll the status word. A tripped safety
switch leaves the gripper dead until tool power is cycled, with every command silently discarded meanwhile.
That is why `connect()` refuses on it: a message now rather than a dropped part later.

---

## 5. Digital-I/O grippers (`jaw_io`, `vacuum`)

The logic is exercised against a fake I/O port, and the controller pins were measured switching against real
controller software. The wiring never touched hardware; measuring it is what the bench tool is for.

This is the family where the numbers are yours: which pin, which bank, active high or low, how long the
cylinder takes. All of it is config on purpose, so the drivers work with whatever hardware you chose, and
that only pays off if the numbers are cheap to measure.

### The bench

```bash
python -m src.robot.drivers.ur --read                         # read every pin, change nothing
python -m src.robot.drivers.ur --watch 0 --for 15             # trip a sensor by hand, see it
python -m src.robot.drivers.ur --measure 4=1 --watch 0 --yes  # the number you came for
python -m src.robot.drivers.ur --where                        # the joints as JointPositions.deg(...), and the TCP
```

Name your cell with `--profile NAME` or `WILLY_PROFILE`. Without either the bench loads the base tree,
which names a Robotiq and no I/O hand, and every refusal prints the profile chain it read so that is
visible at once. `--where` prints a line to paste into a program, for a look pose for instance:
`JointPositions.deg(-45.0, -100.2, -110.0, -60.0, 90.0, 0.0)`.

`--measure` drives an output, times how long the watched input takes to answer, and prints the milliseconds
that become `close_settle_s` and `engage_timeout_s`. It never moves the arm. Run it a few times and
configure the worst reading: the driver waits that budget out, so a typical value becomes a dropped part on a
slow stroke.

Every write is gated behind `--yes`, and in a non-interactive shell the gate refuses rather than prompts: a
prompt that reads end-of-file and takes the default is how an unattended script drives a coil nobody
authorised. From Python the gate is a required `confirm` argument with no default (section 6).

Take `--port` seriously. Standard output 4 is in the control cabinet; the wrist has no tool output 4. A
tool-mounted gripper is usually on `tool`, which both config blocks default to, and the UR tool connector
has two digital outputs and two inputs, 0 and 1 each: with `io_port: tool` the load refuses any pin above 1
in `jaw_io` and in `vacuum`. A pin number measured in the control box needs `io_port: standard` or
`configurable`.

### What `connect()` does depends on your wiring

`jaw_io` is the one gripper whose connect behaviour follows from what you wired:

| Your cell | What `connect()` does |
|---|---|
| end-stop switches wired | reads them; **opens only on an empty reading**, holds a detected part and warns so a person decides |
| no switches, `open_on_connect_without_feedback: true` | **opens unconditionally**, and anything held is dropped (refused for `single_toggle`) |
| no switches, flag off | does not actuate at all |
| `confirm_open_at_start: true` on a solenoid | first **asks** whether the jaws stand open, as a toggle does, then the row above that fits |
| `single_toggle` | **asks** whether the jaws stand open, before anything moves; closed is answered with one pulse to open them or an abort; with no terminal the connect is refused |

### A single toggle asks where its jaws stand

`single_toggle` is one output where every pulse flips the jaws, and nothing is read back: the schema
refuses a feedback input on it, and `confirm_open_at_start: false`. The owner's Hand-E on the Robotiq
I/O Coupling is one. So the program counts its own pulses, and only a person can say where the count
starts. The gripper's connect, which a program makes once at its start after the arm connects and
before anything moves, asks at the terminal:

```text
The jaws of the single_toggle hand on tool output 0: every pulse flips them and nothing reads them back, ...
Look at them. Do they stand OPEN? [Enter or 'open' = open, 'closed' = closed]:
They stand CLOSED. [p] open them now with one pulse on tool output 0, which releases anything between them; [a] abort:
```

Enter or `open` starts the count open. `closed` offers `p`, one pulse there and then, after which the
stroke (`close_settle_s`) is waited out, or `a`, which refuses the connect and rolls the arm back.
With no terminal, and no question handed to the driver (`JawIOGripper(..., ask=...)`), the connect is
refused in one line. Nothing is kept between programs: the next one asks again, so a pulse from the
pendant's I/O tab or a bench `--pulse` between programs costs nothing.

Only a key pressed after a question is shown answers it. Before each question at the terminal the
console's typeahead is discarded: an Enter pressed while the models loaded, or meant as push-to-talk in
examples 12 and 13, used to answer the connect question unseen, and with the jaws really closed every
command after it ran inverted. A question handed in with `ask=` owns its input and is not drained. The
hand counts as connected only once the answer is in and any pulse it chose has gone out: while the
question waits, every command is refused, and a disconnect from another thread refuses the connect.

From there every pulse flips the count, whatever verb sent it:

| Verb | Pulses |
|---|---|
| a pick, before the arm moves | **none**, whatever `pre_open_mm` says; where the count says closed (a pick with no place before it) or cannot say, the pick asks again, and with nobody to ask it is refused before any motion |
| a pick, at the part | exactly one, then `close_settle_s` |
| a place or a release | one where the count says closed, then `close_settle_s`; none where it says open, and the report says `already open` |

At a pick start the question takes a word, not Enter:

```text
Look at them. Do they stand OPEN? [type 'open' or 'closed'; Enter alone is no answer here]:
```

The program already believes the jaws stand closed there, so an empty line is asked again ("An empty
line is no answer here."), and three answers that are none of the choices refuse the pick. A question
whose connection changed while it waited, a disconnect or a reconnect from another thread, is refused
without a pulse, and its answer is not acted on.

The count flips only once the pin has read back HIGH, not when the write returns: a write the
controller accepted says nothing about the pin (a wrong bank, a reserved pin). The driver reads the
output back every 8 ms, one CB3 cycle, for about `pulse_s` and two cycles, 50 ms at least. A pin that
never reads HIGH flips nothing, the low is still sent, and the command raises, naming the bank and the
pin: the output never read HIGH, so nobody can say whether the jaws flipped. No further pulse goes out
until the next pick's question,
or the next connect's, has a person say where the jaws stand. A pin that still reads HIGH before a pulse
would give no edge, so nothing is sent. The solenoids read their writes back the same way and raise
where a level or a coil never shows.

A toggle takes no width: `set_width_mm` is refused, the hand verbs and the pick loop say open or close,
and the load does not ask `closed_below_mm` to sit between the widths. Nothing measures the jaws, so a
pick counts as grasped and its report says `hold not checked (no sensor)`, with no millimetres.

A pulse the count never saw inverts every later command: one lost to an e-stop or a cable, a power cut
mid stroke. Nothing on this wiring can notice it; the next program's question is where a person puts it
right. Measure the pulse the device needs with
`python -m src.robot.drivers.ur --profile NAME --pulse 0 --for 0.2 --yes`: one call must flip the jaws
once. The load refuses a `pulse_s` under 0.05 s for `single_toggle` and `double_solenoid`, several
controller cycles, and the desk warns (`toggle pulse`) while a toggle's is under 0.1 s. `--jaws open`
(or `closed`) moves them through the driver, which asks the question first. Never move them by hand: a
device that keeps its own flip state is not moved by a hand pushing its jaws open.

### The travel time

Where no switch is read, `close_settle_s` is the only wait there is, on a close and on an open: the
arm moves on that long after the edge, whether the jaws have arrived or not. Its schema default,
0.3 s, is a small cylinder's stroke, and a Hand-E on its I/O coupling can take about 2 s. Time the
full stroke both ways, with `--measure` where an input answers it or by stopwatch or video, and set
`close_settle_s` to the longer, with a margin. The desk warns (`jaw travel time`) while it is still the
default.

**`closed_below_mm` reads only the widths you send yourself.** A solenoid jaw closes at or below it
and opens above it when it is given a width (`set_width_mm`). The hand verbs (`pick`, `place`, `grasp`,
`release`) and the pick loop tell `jaw_io` open or close by what they mean instead, so no width turns a
verb round. For a solenoid the load refuses a value at or above `max_width_mm` or below `min_width_mm`.

`vacuum` differs on purpose: it asserts off on connect, because releasing a latched cup is cheap, while
opening jaws that hold a rigid part drops it wherever the arm stands. Its `disconnect()` never raises,
because a teardown path must not strand a held part.

### The feedback is the real payoff

With a vacuum switch wired, `is_object_detected()` is a measurement rather than the commanded state. The jaw
driver answers the same question from whatever is wired, in a fixed order. A dedicated part-present pin wins
outright. Failing that, a fully-open **and** a fully-closed reed switch let it infer the answer: "fully
closed" means the jaws met each other and the grasp is empty, and "stopped in between" means something is
between them. Failing both, it returns the **commanded** state. Wire nothing and you get no measurement,
only a restatement of what you asked for.

---

## 6. From Python

Every gripper satisfies the same vendor-neutral `Gripper` Protocol, so a caller that holds one does not know
which driver it has. `Robot` builds the arm and the gripper your config names, with no pick service around
them, and connects them in the order a cell connects (the snippet at the top of this page).

Load your cell's profile: the base tree names no hand, and a robot is not built on a hand nobody named. At a
desk, `load_tree("console_dummy")` builds a dummy arm with a dummy hand. A gripper that had to be
substituted is refused at `connected()` with `NoRealGripper`, as the cell refuses it (section 7).
`Robot.from_tree(tree, gripper=None)` builds the arm alone.

`grasp` and `release` report what was measured, not what was commanded:

- `report.hold` is `HELD` or `EMPTY` where the gripper measured it (the Robotiq's gOBJ, the OnRobot status
  word, a vacuum switch, a part sensor or reed switches), and `UNMEASURED` where nothing could.
- `report.width_measured` says whether the width was read back from a position sensor.
- A held or unmeasured grasp is modelled as a carried part on an arm that models one. `report.payload` reads
  `ATTACHED` on a cuRobo UR that declares `safety.planning_world.payload.length_mm`, `FILTER_ONLY` when the
  planner declined it, and `NOT_MODELLED` with the reason otherwise. A measured empty close attaches
  nothing.
- A release that still measures a part reads `RELEASE_NOT_CONFIRMED` and keeps the model.
- No force is commanded. A gripper that raises is stopped once and reported as `GRIPPER_FAULT`, and
  `robot.is_holding()` reads the hold without commanding anything.

A pick and a place are one call each, for a part whose pose you already know in BASE
([`examples/real_robot/05_pick_and_place_a_known_part.py`](../../examples/real_robot/05_pick_and_place_a_known_part.py)):

```python
from willy import Pose, Robot, load_tree

robot = Robot.from_tree(load_tree())
part = Pose.tool_down(450.0, 100.0, 120.0, yaw_deg=90.0)   # millimetres in BASE, inside your workspace
tray = Pose.tool_down(300.0, -250.0, 140.0)
bench = "a known part on a clear table, no camera"

with robot.connected():
    picked = robot.pick(part, 40.0, decline=bench)   # standoff, a line in, close, a line out
    print(picked)
    if picked.ok:
        print(robot.place(tray, decline=bench))
```

The pose's own +Z is the approach. `pick` opens the jaws to the hand's width, makes a planned move to a
standoff `standoff_mm` (80) back along the approach, drives a straight line to the pose, closes to
`width_mm` less `squeeze_mm` (1), and drives the line back out. `place` does the same with a release.

Before any command both verbs refuse a robot with no usable gripper, a closed link, a pose not in BASE, a
camera world the arm would refuse the motion for, and an arm whose motions do not go through cuRobo and the
exact mesh guard. A cuRobo UR or simulator arm judges every sample of the line before it runs; the dummy
and the simulator mock set the pose; an ik UR, a KUKA and a simulator on ik or RMPflow are refused. A
refused motion ends the verb with nothing commanded after it, and so does a camera that could not vouch for
the cell, as the outcome `CAMERA_WORLD_UNAVAILABLE`. A close that measures nothing opens and backs out, and
a release the gripper does not confirm leaves the arm where it stands. `camera_world=CameraWorldDecline("<why>")`
is a second spelling of `decline="<why>"` and works the same.

`robot.move(pose)`, `robot.move_joints(joints)` and `robot.home()` move the arm alone, with the same
refusals and the same `decline=`, through the arm's own checked verbs. Each returns a `MotionReport` that
prints the outcome, the arm's status and message, the camera-world stamp and the route; on a desk arm the
route and the stamp say `UNPLANNED`.

On a cell whose cameras are handed in, `Robot.from_tree(tree, cameras=[camera])`, leave `decline` unset:
every motion plans against the live world, and a declined motion is refused there. Pass the target's
`SegmentationOffer` as `keep_out=` (a `Locator` result's `keep_out(i)` gives one), and the part is held out
of the planner world through every motion of the pick
([`examples/real_robot/13_speak_pick_and_hand_handover.py`](../../examples/real_robot/13_speak_pick_and_hand_handover.py)). The report
carries the camera-world stamp of each motion, the line reading taken before the first one, and the hand
report.

The bench is a noun too, and its interlock is not optional:

```python
from willy import Robot, load_tree
from src.robot.core.arm_capabilities import SupportsDigitalIO
from src.robot.drivers.ur.bench import Bench, Measure

tree = load_tree()
robot = Robot.from_tree(tree, gripper=None)            # the arm alone; the bench switches its pins
assert isinstance(robot.arm, SupportsDigitalIO)
bench = Bench.from_robot_config(tree.robot, robot.arm, confirm=lambda what: input(f"{what}? ") == "yes")
with robot.connected():
    print(bench.run(Measure(pin=4, level=True, watching=0)).render())
```

`confirm` has no default. The functions underneath have no gate at all and energise a pin the moment they
are called, so a default that returned `True` would be the absence of an interlock wearing its name.
`Bench.from_robot_config` also refuses to guess the bank: a config whose gripper is neither `jaw_io` nor
`vacuum` declares no bank, and the cell then has no digital-I/O gripper to exercise.

One action at a time. `Bench.run` takes a single action object, `Read`, `Watch`, `Set`, `Pulse` or
`Measure`, so "read and also set" cannot be expressed, and the command line refuses the same combination
rather than honouring one flag and dropping the other.

---

## 7. When it does not work

| Symptom | Most likely |
|---|---|
| The connect is refused with `NoRealGripper` | a **substituted `NullGripper`**: the end-effector you named could not be built. The refusal names the reason and the fix |
| Robotiq: connect works, every read times out | the URCap daemon is up and its RS-485 link to the gripper is dead |
| Robotiq: the connection dies on the first byte | nothing listens behind a forwarded port: the URCap is not installed or not running |
| OnRobot: silence | silence is the documented symptom of *every* misconfiguration: wrong unit id, Modbus not served, no tool on that side |
| Digital I/O: the write is accepted and the pin does not move | wrong bank; try `--port standard`, `configurable` or `tool` |
| Digital I/O: the output was driven and the input never answered | wrong pin, an unpowered sensor, or an active-low switch |

**The first symptom has its own paragraph.** A misconfigured end-effector is built as a `NullGripper`
carrying a `GripperSubstitution`, with one reason per case: an unrecognised vendor name; `robotiq` on an
arm that is not a UR, or on a UR carrying no controller address; `vacuum` on an arm with no digital I/O;
`jaw_io` on an arm with no digital I/O; a recognised name with no driver. `onrobot` has no such
precondition, because the Compute Box does not depend on the arm. The record carries the reason, what was
requested, a sentence an operator can act on, and the fix.

`connect_cell` reads the substitution record before it commands the arm and raises `NoRealGripper`, so the
command line prints `[connect] FAILED: NoRealGripper: ...` and exits 1, and `Cell.connected()` and
`Robot.connected()` raise. The operator console refuses the same connect with `403 no_real_gripper` and
reaches the hardware through the same function, so every entry point gives one answer with one sentence.
Without this refusal the cell would come up, every pick would report success, and the jaws would close on
nothing.

The guard is keyed on the substitution, not on the absence of jaws. `gripper.vendor: none` carries no
substitution record: the operator said this cell has no end-effector, and a calibration rig or a
camera-only bring-up is a legitimate cell. It connects. With `robot.grasping.verification` enabled,
`WidthDeltaGripperVerifier` refuses both cases by name: a substituted gripper as `FAILED` /
`no_end_effector_built`, since it answers `get_width_mm()` with its configured maximum whatever it was
commanded, and `gripper.vendor: none` as `no_end_effector_configured`. Verification is **off** in the
shipped tree, so on the default open-loop pick nothing reads a width, which is why the refusal sits at the
connect.

---

## 8. What is not here

- **No three-finger gripper, no vacuum URCap tools, no 2FG7.** Each is a different protocol or an unsourced
  register map. A vacuum URCap is the dangerous near-miss: it installs a *different* URCap and reuses the
  same variable names with different meanings, where `POS` is a vacuum level.
- **No Schunk or Franka driver.** Both are declared names with no implementation. A Schunk on the tool I/O
  works through `jaw_io`. In simulation the EGU-50 is `robot.gripper.model: schunk_egu50`, mounted standalone
  and driven by the vendor-neutral simulated jaw gripper through its profile.
- **No force or current reading from any real driver.**
- **No gripper driver has touched hardware.** The bench, the probes and the read-only rungs exist because the
  measurement is yours to make.

## Where to look next

- [`src/robot/grippers/README.md`](../../src/robot/grippers/README.md): every driver, the substitution rules
  and the simulator profiles
- [`src/robot/core/README.md`](../../src/robot/core/README.md): the `Gripper` and `ObjectDetectingGripper`
  contracts
- [`src/robot/drivers/ur/README.md`](../../src/robot/drivers/ur/README.md): the arm that owns the digital
  I/O these drivers switch
- [`scripts/ursim/README.md`](../../scripts/ursim/README.md): the probes in the order they are meant to run
- [hande_gripper_bringup.md](../runbooks/hande_gripper_bringup.md) and
  [your_own_gripper.md](../runbooks/your_own_gripper.md): a Hand-E, and a hand this repository does not ship
- [04](04-robot-and-safety.md) for declaring one, [05](05-pick-loop.md) for the pick that closes it
