# Grippers (`src/robot/grippers`)

The drivers for the hand on the arm, each behind the `Gripper` Protocol. Mostly you interact with them through the `Robot` interface rather than directly.

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
`print(robot)` says which hand the tree really built, or what stands in for it and why, before anything connects.
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
| `jaw_io` | `JawIOGripper` | one or two output pins (one for a toggle, where every change of it moves the jaws), optional reed switches or a part sensor | an arm with digital I/O: the UR driver |
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
The hand verbs read these two, never the command echoed back. A push reads a measured width before its
first motion: within 2 mm of fully open it pushes, and a gripper that measures its width found **not
connected** there, or whose width read fails, ends the pick as a gripper fault, with nothing moved (the
owner, 2026-09-30). The Robotiq driver counts itself connected until the program disconnects it, so a
socket that stopped answering shows as a read that fails.

| Driver | A hold is read from | Width |
|---|---|---|
| `robotiq` | gOBJ: fingers stalled on something, or reached their target | measured |
| `onrobot` | the grip-detected bit of the status word | measured |
| `jaw_io` | the reed switches or a part sensor, where wired; nothing on a `single_toggle` | commanded; none on a `single_toggle` |
| `vacuum` | the vacuum switch, where wired | commanded |
| simulated suction | the physical bond in Isaac | commanded |
| `IsaacGripper` | nothing | measured outside mock mode |
| `dummy`, `none` | nothing | commanded |

Connecting differs on purpose. `vacuum` switches suction off at connect, because releasing a cup left
latched is cheap. A `jaw_io` hand left closed may hold a rigid part, so it opens only when feedback says
it is empty, holds and warns when feedback says a part is there, and does not move at all with no
feedback wired, unless `open_on_connect_without_feedback` is set, or asks a person first where it opted
into `confirm_open_at_start`.

A `single_toggle` (the owner's Hand-E on the Robotiq I/O Coupling, one tool output, no feedback) reads
no sensor at all, and a feedback input on it is refused. Every change of its output moves its jaws,
switched on as much as switched off (the owner at the pendant, 2026-09-28), so a command is ONE change,
left where it went, and nothing is pulsed: the low, high, low pulse it sent before moved the jaws two
or three times. The program counts its own changes from where a person says they stand: its
`connect()`, once per program start and before anything moves, reads the output without writing it and
asks at the terminal whether the jaws stand open (Enter or `open` = open), or through the seam a console
installed ([below](#the-jaws-question-through-a-seam)), and a person who says closed
chooses one change to open them or an abort. With no terminal and no `ask=` handed in, the connect is
refused. The hand counts as connected only once the answer is in, so
nothing can command it while the question waits. Nothing is kept between programs. It is a
`core.gripper.TogglesWithoutSensor`, which is how the pick code knows, without importing this package,
to send no change before the arm moves (it asks `jaws_open_for_a_pick()`, which asks the person again
where the count says closed or cannot say, and there takes only the word `open` or `closed`: an empty
line is asked again), exactly one at the part and one at a release that finds the jaws closed. The
output is read before every command: one that no longer stands where the program left it was switched
by hand, at the pendant above all, and the command is refused with nothing sent; the next pick asks. It takes
no width (`set_width_mm` is refused), and a pick with it counts as grasped with the hold not checked,
because there is no sensor. The bench's `--jaws open|closed` drives the jaws through the driver, which
asks first. A push (recovery's `nudge_target`, on a wrist camera's pick in `dense_clutter` only) only
reads the hand: it runs on a connected toggle whose count says open (`jaws_closed` false,
`why_jaws_unknown()` empty), writes no output and never calls `jaws_open_for_a_pick`. A count that says
closed means no push. One nobody can vouch for when the push reads it, once its plan and budgets passed
(the output switched at the pendant mid-pick, the count lost or unreadable), ends the pick as a gripper
fault, as the pick-start refusal does. The push reads it again before each contact leg and once the arm
is up, and one nobody can vouch for there stops the push where the arm stands, a person to decide. The push
asks nobody either way. A push refused before that read falls through, and `next_target` reads the count
before it drives a wrist pick's looks again: one nobody can vouch for ends the pick as a gripper fault before
any look is driven (`core.gripper.why_toggle_count_unknown`, read before each contact leg, once the arm is
up and before the looks run again; the push's first read applies the same rule).

At the terminal the console's typeahead is discarded before each question (`msvcrt` on Windows,
`termios.tcflush` on POSIX), so an Enter pressed earlier cannot answer it; an `ask=` handed in is not
drained. A question whose connection changed while it waited (a disconnect or another connect) is
refused with nothing sent. Every write is read back (`get_digital_output`, every 8 ms: for a solenoid
for about `pulse_s` and two controller cycles, 50 ms at least, for a toggle 0.25 s): a toggle's count
flips only once its output reads the new level, and one that never does raises and leaves the count
unknowable until a person says where the jaws stand; a solenoid raises where its level or coil never
shows. The config refuses a `pulse_s` under 0.05 s for `double_solenoid` (a toggle never reads it),
and a pin above 1 on `io_port: tool`, which has outputs 0-1 and inputs 0-1 (`jaw_io` and `vacuum`
alike).

The hand verbs (`pick`, `place`, `grasp`, `release`) and the pick loop tell `jaw_io` open or close by
intent (`OpensAndCloses.set_closed`), so `closed_below_mm` cannot turn a verb round; on a solenoid it
only reads the widths a caller sends through `set_width_mm` itself. Both touch the controller's I/O at connect, so the arm connects first; `Robot` does that for
you.

**The stroke, left to the verb.** `set_closed(closed, wait=False)` sends the same one change and comes back at once
with the moment the jaws' stroke is over (`time.monotonic()` seconds, `close_settle_s` on); `wait_settled(deadline)`
sleeps what is left of it. In between the verb judges its next leg and sends nothing (`robot.motion.judge_next_leg`,
the owner, 2026-10-09): the grasp's carry while the jaws close, the line out of a place while they open. Only a wait
that is the travel time alone is left so, a toggle's and a solenoid's with no switch to read; a close that reads its
reeds waits here as ever and returns `None`, and so does a command that moved nothing. A command that comes while a
stroke left so still runs waits it out first, as it waited inside `set_closed` before, so a change never turns the
jaws round mid-stroke. `wait=True`, the default, is the command it always was.

## The jaws question, through a seam

The operator console asks a toggle's question in the browser, never at the terminal of the server it runs in.
It hands the hand a **structured seam**, and every question goes there from then on:

```python
from src.robot.grippers.jaw_io import JawAsking


def ask(question: JawAsking) -> str:
    # question.stage: "where" (choices open, closed) or "open_now" (choices open_now, abort): after "closed", or
    # first and only at a check where the program's own count says the jaws stand closed
    return "open"           # one of question.choices; raise EOFError for no answer


gripper.answer_questions_with(ask, before_change=lambda: "")   # "" lets the one change out
```

- **`JawAsking`** carries the `stage`, the `choices` in the console's words, whether a connect asks, the reason,
  where the output is (`tool output 0`, also `where_pin`, a lock-free read), the hand's own text, the attempt
  of three and why it is asked again. There is **no default**: an empty answer is no answer, at a connect too.
  The seam takes only the offered words (and `p` and `a` for the open-now stage); anything else is asked again,
  saying why, and `EOFError` is no answer. A question nobody answered is refused, never taken for "open".
- **A person's "closed" is the count from then on**, before the open-now question: an abort after it leaves
  the count CLOSED, never unknown.
- **`before_change`** is asked under the hand's lock immediately before the one change "open now" sends. Only
  the empty string lets it out; any other answer, no sentence at all, or a raise refuses it with nothing sent,
  and the refusal names it. The connection is read again right after the gate, so a disconnect that landed
  while the gate read the controller refuses too, and an output switched at the pendant meanwhile raises with
  nothing sent. The gate comes and goes with the seam: `before_change` is read-only on the hand.
- **`confirm_where_the_jaws_stand(reason, before_change=...)`** is the check before the arm moves again after
  a stop: it **always asks**, also where the count says open, and answers `""` once the jaws stand open, else
  the refusal. A hand that is not a toggle answers `""` at once.
- **A count that says closed is never answered "open".** Where the program itself knows the jaws stand closed
  (its count says CLOSED, no change failed or went unread since, and the output still reads the level that
  count left), the check does not ask where they stand: an "open" against that evidence would turn the count
  round with nothing sent, and the next close would open them. It asks only `open_now` or `abort`, and an
  abort keeps the count CLOSED. Where the count cannot vouch for itself (the output switched at the pendant, a
  change that failed) or says open, it asks `where`, taking only `open` and `closed`.
- `question_seam_installed()` says whether a seam asks; the console refuses to connect a toggle without one.

**A pulse a halt cuts short still ends low.** A halted arm refuses every output write with `ArmHalted`, the
toggle's included, and sends nothing. A double solenoid's coil and a vacuum blow-off are different: they are
pulsed high and must go back low however the pulse ends, so they end it through `end_pulse`, which falls back to
the arm's `end_output_pulse`, the one write a halted arm makes, only ever low. A toggle hand never uses it.

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
| `jaw_io` on a real hand: a Hand-E switched as `single_toggle` on tool DO0 of a UR10 (CB3) | run on a physical cell |
| The jaws question answered in the browser, "open now" as one DO0 edge, no DO0 edge after a halt | measured against real controller software: URSim CB3, `scripts/ursim/probe_console_task.py` and `probe_halt.py` |
| The Robotiq driver on a real Hand-E over its URCap socket | run on a physical cell: measured with a UR; on the UR10 (CB3) above the socket on port 63352 was refused, so its Hand-E runs as `jaw_io` |
| OnRobot and `vacuum` on a real hand | never touched hardware |

In the suite the Robotiq arithmetic runs against a fake driver, and it is the one path URSim cannot
cover, because the URCap opens port 63352. The OnRobot driver runs against a fake Modbus client, and the
I/O drivers against a fake I/O port. None of the real drivers reports force or current.

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
