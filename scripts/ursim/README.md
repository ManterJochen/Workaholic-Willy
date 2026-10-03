# URSim probes (`scripts/ursim/`)

URSim is Universal Robots' offline simulator: the real controller software (PolyScope and the
URControl core) driving a robot that does not exist. The scripts here start it and then prove, one
layer at a time, that this stack talks to a UR controller: the protocol, the safety modes, the I/O
registers, **"halt now"**, and the operator console's task with its way back after a stop. They prove
nothing about the wiring of a real cell.

You need Docker inside WSL2 and Python with this repository installed. The Docker install and the
controller's traps are in the UR section of
[`docs/runbooks/cell_bringup.md`](../../docs/runbooks/cell_bringup.md#ur).

```bash
# inside the WSL distro that runs Docker, from the repository root
bash scripts/ursim/ursim.sh up UR5        # or UR3; URSIM_FRESH=1 for a clean controller
# then enable Remote Control at http://localhost:6080/vnc.html (Settings, System, Remote Control)
# and switch the selector at the top right from Local to Remote, after every start

# then, from the repository root, with this repository's Python
python scripts/ursim/probe_ursim.py                       # the SDK talks to the controller
python scripts/ursim/probe_our_driver.py --profile ursim  # this repository's driver does
```

> [!WARNING]
> Every probe that commands motion reads the controller address from a profile: `--profile`, else
> `WILLY_PROFILE`, else a URSim profile. `probe_curobo_bed.py` reads `WILLY_PROFILE` alone. With
> `WILLY_PROFILE` naming a real cell they move the real arm, so run them where it is unset or names a
> URSim profile, and pass `--profile ursim` where the script takes it.

## Run them in this order

| Step | Script | What it proves | Moves |
|---|---|---|---|
| 0 | [`ursim.sh`](ursim.sh) `up\|down\|status MODEL` | the simulator runs and its dashboard answers; `MODEL` is `UR5` or `UR3` | nothing |
| 1 | [`probe_ursim.py`](probe_ursim.py) | `ur_rtde` talks to the controller, with none of this repository's code | power, brakes, I/O outputs |
| 2 | [`probe_our_driver.py`](probe_our_driver.py) | `URRobotArm`: payload push, tool frame check, `RobotStatus`, a joint move through `SafetyPreflight` | the arm; I/O once confirmed |
| 3 | [`probe_grippers.py`](probe_grippers.py) | `JawIOGripper` and `VacuumGripper` reach the controller's digital I/O | I/O once confirmed |
| 4 | [`probe_robotiq_urcap.py`](probe_robotiq_urcap.py) | whether a Robotiq URCap answers on its socket, and which way its position runs | nothing |
| 5 | [`probe_protective_stop.py`](probe_protective_stop.py) | what the driver reports when the controller enters a protective stop | the arm |
| 6 | [`probe_pickloop_stop.py`](probe_pickloop_stop.py) | the same stop against the pick loop's verdict | the arm |
| 7 | [`probe_halt.py`](probe_halt.py) | "halt now", M0 to M9 and M3b: the latch, the brake's latency and stop point, 200 watched moves with no early return, a braked path that sends no later waypoint, no DO0 change after a halt, a protective stop (M7) and the latch across a reconnect | the arm; tool DO0 |
| 8 | [`probe_console_task.py`](probe_console_task.py) | the console's task, C1 to C5: three cycles with exactly 2 DO0 edges each, halt in the approach and in the carry, a protective stop, Disconnect during the planner start, and the way back through "the cell is clear", the jaws question and Restart | the arm; tool DO0 |
| | [`watch_stop.py`](watch_stop.py) | what the stack sees when a person presses the stop at the pendant | three small joint moves after the stop |
| | [`probe_base_frame.py`](probe_base_frame.py) | whether the controller reports poses in the DH base frame or the one turned half a turn | nothing |
| | [`probe_curobo_bed.py`](probe_curobo_bed.py) | the checked motions with the cuRobo planner on, and what each check costs | the arm |
| | [`probe_push_on_the_mat.py`](probe_push_on_the_mat.py) | the push and clearing a blocker, with a cell's own tree, a recorded mat and the toggle hand | the arm and the tool output |

Each script's docstring holds its procedure and its exit codes. Most exit 0 when every step answered
as it should, 1 when a step failed or nothing could be measured, and 2 when no controller was
reachable or the profile is not a UR cell. `probe_base_frame.py` answers 0 for the DH frame, 1 for the
turned frame, 3 for neither and 2 when it cannot read the controller. `probe_curobo_bed.py` prints its
measurements as one JSON record.

How each one is aimed:

- `probe_ursim.py` is fixed at 127.0.0.1 in its source, because it writes standard and tool outputs
  without asking. A flag would make "point it at the real controller" a typo.
- `probe_our_driver.py` and `probe_grippers.py` write a digital output only when it is confirmed: with
  `--yes`, or by typing `yes` at the prompt in a terminal. Without a terminal and without `--yes` they
  refuse each write and say so. The joint move in `probe_our_driver.py` (wrist 3 by 0.15 rad and back)
  is checked by `SafetyPreflight`, not by the flag.
- `probe_robotiq_urcap.py` takes a host as a positional argument, and `probe_base_frame.py` takes
  `--host` (both default to 127.0.0.1). Both only read, so pointing them at a real controller is safe.
- `probe_curobo_bed.py` defaults to `WILLY_PROFILE=ursim,ursim_curobo` and needs the cuRobo environment
  installed on this machine. `WILLY_BED_START_JOINTS`, six radians separated by commas, is a start pose
  it drives to first.

For a UR3e controller, start it with `ursim.sh up UR3` and pass `--profile ursim,ursim_ur3`.

Step 1 is not redundant next to step 2. When step 2 fails, step 1 says whether the fault is in this
stack or in the SDK.

## Halt now and the console's task, on a CB3

Both probes run on 127.0.0.1 only and refuse a profile whose `ur.ip` is not the loopback: they move the arm
and switch tool DO0 without asking. Each writes its own scratch profile layer into a temporary copy of
`config/` (`brake_on_halt: true`, the Hand-E as `jaw_io` `single_toggle` on tool DO0, a declared
`payload.length_mm`), so no shipped profile changes.

```bash
python scripts/ursim/probe_halt.py                        # M0-M6 (M3b included) and M9
python scripts/ursim/probe_halt.py --only M7,M8           # the two that stop the controller's program
# the arm to the console probe's home first, [-90, -90, -100, -80, 90, 0] deg (the pendant's Move tab, or this)
python -c "import math, rtde_control; c = rtde_control.RTDEControlInterface('127.0.0.1'); c.moveJ([math.radians(d) for d in (-90, -90, -100, -80, 90, 0)], 0.6, 0.8); c.stopScript()"
python scripts/ursim/probe_console_task.py --json c.json  # C5, then C1-C4; needs the cuRobo environment
```

- **`probe_halt.py`** exits 0 when every check that ran passed, 1 when one failed, 2 with no loopback UR
  reachable. `--only` picks checks, `--moves` and `--lines` size M3 and M3b, `--json` keeps the numbers. M7
  needs a real protective stop, which it asks of the controller (`triggerProtectiveStop`).
- **`probe_console_task.py`** drives the console itself, in this process: the routes through `TestClient`,
  the real run registry, the URSim UR10 planned by cuRobo, the jaws question answered through the browser's
  route, DO0's edges read on a second RTDE connection. It exits 0 when every check passed, 1 when one failed,
  and 2 with no controller, not the loopback, a cell that did not come up, an arm outside the joint window, or
  **no simulator proven**: it runs
  only where the console says `controller_is_simulator`, on a CB3 too: `ur_rtde` reads no serial below
  PolyScope 5.6, so the driver asks the dashboard's "get serial number" itself (URSim CB3 answers
  `2018309999`). `--cycles` sets C1's tasks, `--model` the arm's
  layer (`ur10`). **It checks where the arm starts** before anything moves, with the console's own joint guard:
  `probe_halt.py` leaves wrist 2 at -90 deg, half a turn from the probe's home and outside the UR10 profile's
  joint window about it, where every path would be refused at its first sample (C1 `failed_in_a_row`, and its
  stop record refusing C2 to C4), so the probe exits 2 there, naming the axis. Move the arm to the probe's home,
  [-90, -90, -100, -80, 90, 0] deg, first: the line above does it on the simulator, with no check of this
  repository on the way. C3 answers the jaws question "open now" alone: a count that says closed is asked
  nothing else since 2026-10-02.
- **`probe_protective_stop.py`** refuses a model it has no measured height for, every CB3 model included: a
  CB3 UR10's IK finds no tool-down pose on the base axis, and the failed IK ends the control script. Use
  `probe_halt.py --only M7` there.

**A CB3 comes up differently** from the e-Series image:
`URSIM_IMAGE=universalrobots/ursim_cb3:latest URSIM_NAME=ursim_cb3 URSIM_FRESH=1 bash ursim.sh up UR10`. It
has no Remote/Local switch. A "Power off" notice comes first (click **Not now**); power on and release the
brakes from the dashboard; only then does "Confirm Safety Configuration" appear, and RTDE refuses until it is
confirmed. After a protective stop the control script has ended, and the first script upload after the
release can time out once. The measurements of both probes, and what they leave for the cell, are in
[`docs/runbooks/console_at_the_cell.md`](../../docs/runbooks/console_at_the_cell.md).

## The push on a cell's own mat

[`probe_push_on_the_mat.py`](probe_push_on_the_mat.py) is the push's simulation gate. It runs whole picks of a cell's
own service against URSim: the tree's arm, its toggle hand, its tool frame, its looks, the guard's distances, the
cuRobo planner and the camera world. It copies the tree into a new folder first and adds one layer, `ursim_mat`, which
sets `robot.ur.ip` to 127.0.0.1 and nothing else. A chain that does not name that layer, a chain whose address resolves
to anything else, and a work folder that is not empty are refused before anything connects.

```bash
# URSim up as a UR10, powered on, brakes released, at the cell's home; then, through the GPU runner
python <gpu runner> scripts/ursim/probe_push_on_the_mat.py --tree <cell>/config --work <new folder> \
    --profile <the cell's layers>,ursim_mat --calibration <the rig's artifact on this box>
python scripts/ursim/probe_push_on_the_mat.py ... --offline      # step 0 alone: nothing connects
```

URSim has no camera and moves no part, so two things stand in, and nothing else:

- **The camera.** [`_mat_scene.py`](_mat_scene.py) holds one recorded look of the owner's mat (the depth crop of
  the first Zollstock pick of 2026-10-01, `tests/data/cell_2026_10_01`) as points, the pile cut out and the mat
  filled in under it. Every grab casts the bench round it (the bench top, the mat's top and its four sides, the
  floor), keeps the recorded look for the yellow bin alone, and adds the scene's parts, from where the camera stands
  now: the TCP the controller reports, composed with the rig's own CAMERA to TOOL. A look from beside the mat then sees
  its sides and the floor past them, as the owner's camera does, where the recorded look alone ended at its own crop.
  `--recorded-bench` keeps that recorded look alone, as before 2026-10-03. The pick perception and the live planner
  world read the same frame. Nothing nearer than 200 mm reads a depth: on 2026-10-01 the owner's D415 read no depth on
  about half its pixels at the pre-grasp, and 200 mm reads 40 to 60 % there.
- **The parts.** A part the jaws close on is carried with the TCP and set down where they open. After a push the
  part stands where the plan put it. The detector and the segmenter ground the prompt on the render's target mask.

Step 0 renders the push scene from each look, finds the supports and asks `plan_push` as the pick would; it reports
and does not gate. Then the scenarios, each a whole pick, the owner's switch set as each needs it (2026-10-03):

| Scenario | The scene | Held where |
| --- | --- | --- |
| `blocker` | critical parts: a 30 mm block, an L of two 30 mm blocks 28 mm off its -x and +y sides | a block gripped and set down on a free spot, then the part gripped; DO0 closed, open, closed; nothing pushed |
| `push` | not critical: the 40 mm cylinder, a 60 mm block 20 mm off its -x side, shifted 20 mm toward +y | the push comes first and may brush the block; every leg judged; DO0 untouched through it; back to the look, a fresh look, the cylinder gripped |
| `cancel` | the push scene | a stop asked for during the push leg: nothing more commanded, a person decides |
| `pstop` | the push scene | a protective stop during the push leg: the same |
| `refusal` | a 40 mm cube beside the cylinder; asked for alone | no motion toward the part, nothing switched |

`--scene NAME=NEIGHBOURS@GAP@X,Y[@TARGET]` places a scenario's part and its neighbours, each
`name[:side[:gap[:along]]]` (`block60:-x::20` is the push scene's block), `NAME#n` runs a scenario again on another
scene, and `--looks first` hands every pick LOOK[0] alone. `--push-mm` asks for a distance; unset, the cell's own
runs, longer where it frees no direction, as a console run without one does. The tool output is read on a second RTDE
connection and every change of it is counted. The result is `<work>/p_result.json`; exit 0 means every item held, 1
names each that did not, 2 means refused before anything connected. Before each run, put the arm back at the cell's
home and clear a protective stop the last run left (`pstop` leaves one).

`--push-stand-in` is the plan's stand-in for the push scenarios, and the result says where it was used: the
calculator answers the part `ALL_COLLIDED` until a push ran, and `plan_push`, where it refuses on the pick's own
inputs, is asked again with the part's whole surface and the mat round it. The plan's execution, the guard, the world,
the toggle and the stops stay the cell's own. Without it a push plans only where the cell's looks see the part's foot
and the table all round its landing.

## An open port is not a gripper

With port 63352 published by Docker and no URCap installed, a TCP connect succeeds and the connection
drops on the first byte. `probe_robotiq_urcap.py` therefore prints the connect as a non-result, and
only a parseable reply counts. It also settles the one thing the vendor documents do not: which way
the position number runs. Run the probe, note the position, move the fingers from the teach pendant or
the Compute Box web interface, run it again, and see which way the number went.

## What a green run means

The I/O registers exist and read back, and nothing is connected to them. Pin assignment, reed switch
polarity, cylinder travel time and whether a closed jaw holds anything need a bench with the gripper
bolted on, and every probe says so in its output. A green run means the software is right about
itself: measured against real controller software, never touched hardware.

## The container needs a WSL session held open

If the container exits within a minute or so of starting, the cause is the WSL session, not URSim.
Every `wsl -e bash -lc ...` opens a session and closes it again, and with no session open the distro
is torn down with systemd, dockerd and every container in it. Hold one long-lived session open:

```powershell
Start-Process wsl.exe -ArgumentList "-d","<distro>","-e","sleep","infinity" -WindowStyle Hidden
```

A `nohup ... &` inside a throwaway `wsl -e bash -lc` is not enough: it dies with the session. Two
things the early exit is not: memory or disk (`docker inspect` reports `oom=false`), and the
calibration warning in `URControl.log` about `/ursim/.urcontrol/calibration.conf`, which every healthy
run prints because the file is absent from the image too. [`ursim.sh`](ursim.sh) carries the rest of
the procedure in its header: the ports, URCaps, and the WSL idle timeout.
