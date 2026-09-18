# URSim probes (`scripts/ursim/`)

URSim is Universal Robots' offline simulator: the real controller software (PolyScope and the
URControl core) driving a robot that does not exist. The scripts here start it and then prove, one
layer at a time, that this stack talks to a UR controller: the protocol, the safety modes, the I/O
registers. They prove nothing about the wiring of a real cell.

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
| | [`watch_stop.py`](watch_stop.py) | what the stack sees when a person presses the stop at the pendant | three small joint moves after the stop |
| | [`probe_base_frame.py`](probe_base_frame.py) | whether the controller reports poses in the DH base frame or the one turned half a turn | nothing |
| | [`probe_curobo_bed.py`](probe_curobo_bed.py) | the checked motions with the cuRobo planner on, and what each check costs | the arm |

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
