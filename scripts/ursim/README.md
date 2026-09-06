# `scripts/ursim/`: bringing a UR controller up, and proving this stack talks to it

URSim is Universal Robots' offline simulator: the real controller software (PolyScope plus the
URControl core) driving a simulated robot. That makes it the honest middle step between a mock and a
physical cell. It exercises the RTDE protocol, the safety modes and the I/O registers, and it exercises
none of the wiring.

## Run them in this order

| | script | what it proves |
|---|---|---|
| 0 | [`ursim.sh`](ursim.sh) `up\|down\|status MODEL` | the simulator is running and its dashboard answers. `MODEL` is `UR5` or `UR3` |
| 1 | [`probe_ursim.py`](probe_ursim.py) | the SDK talks to a controller: `ur_rtde` from Windows |
| 2 | [`probe_our_driver.py`](probe_our_driver.py) | this repository's `URRobotArm` does: payload push, tool-frame verification, `RobotStatus`, a commanded move through `SafetyPreflight` |
| 3 | [`probe_grippers.py`](probe_grippers.py) | `JawIOGripper` and `VacuumGripper` reach the controller's digital I/O |
| 4 | [`probe_robotiq_urcap.py`](probe_robotiq_urcap.py) | whether a Robotiq URCap daemon answers on the gripper socket, and which way its position number runs. Read-only, and there is no flag that makes it write |
| 5 | [`probe_protective_stop.py`](probe_protective_stop.py) | what the stack does when the controller enters a protective stop |
| 6 | [`probe_pickloop_stop.py`](probe_pickloop_stop.py) | the same, one layer up: a real stop against the pick loop's verdict |
| | [`watch_stop.py`](watch_stop.py) | what the stack sees when a human stops the cell. An observation tool, run once per cell |

How each probe is aimed differs, and the difference is deliberate. `probe_ursim.py` is fixed at the
loopback address in the source, which is its only interlock: it drives the raw SDK and writes I/O, and
a flag would turn "point that at the real controller" into a typo. The probes from step 2 onward take
`--profile` and read the controller address out of the config, so they exercise the same chain a cell
does, and the two that command motion also require `--yes`. Only `probe_robotiq_urcap.py` takes a host
as a positional argument, because it is read-only in the strongest sense and answering for a real
controller is the point.

Step 1 looks redundant next to step 2 and is not. When step 2 fails, the only cheap question is whether
the fault is in this stack or in the SDK, and step 1 is the answer. Keep them separate.

## An open port is not a gripper

`probe_robotiq_urcap.py` exists because a TCP connect proves nothing here. With port 63352 published by
Docker and no URCap installed, the connect succeeds and the connection dies on the first byte: a port
scanner says yes and there is no gripper. Only a parseable reply is evidence, which is why the connect
is printed as a non-result and the reading is the answer.

It is also the instrument for the one question the vendor documentation cannot settle, which is
polarity. The manual gives 0 as open, and the SDK's own header contradicts itself between its unit
systems. So the procedure never asks anyone to trust a number: run the probe, note the position, move
the fingers from the teach pendant or the Compute Box web interface, run it again, and see which way
the number went.

## URSim validates the call path, never the wiring

The I/O registers exist and read back, and nothing is connected to them. Pin assignment, reed-switch
polarity, cylinder travel time and whether a closed jaw actually holds anything need a bench with the
gripper bolted on. Every probe here says so in its own output. A green run means the software is right
about itself.

## The container needs a WSL session held open

If the container exits within a minute or so of starting, the cause is the WSL session and not URSim.
Every `wsl -e bash -lc ...` opens a session and closes it again; with no session left open the distro
is torn down and takes systemd, dockerd and every running container with it. Hold one long-lived
session open:

```powershell
Start-Process wsl.exe -ArgumentList "-d","<distro>","-e","sleep","infinity" -WindowStyle Hidden
```

A `nohup ... &` inside a throwaway `wsl -e bash -lc` is not enough: it dies with the session it was
meant to outlive. A real, persistent `wsl.exe` process is.

Two things the early exit is not, both checked so nobody checks them again. It is not memory or disk
(`docker inspect` reports `oom=false`). And it is not the calibration warning: `URControl.log` reports
`/ursim/.urcontrol/calibration.conf` as corrupt or missing, and that file is absent from the pristine
image too, so every healthy run prints it.

Remote control mode is required on the controller and cannot be set from these scripts.
[`ursim.sh`](ursim.sh) carries the rest of the procedure in its own header.
