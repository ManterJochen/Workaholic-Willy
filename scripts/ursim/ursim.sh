#!/usr/bin/env bash
# Start URSim (Universal Robots' offline simulator) in Docker inside WSL2.
#
#   ursim.sh up UR5                starts, or resumes, a UR5e controller
#   ursim.sh up UR3                starts, or resumes, a UR3e controller
#   URSIM_FRESH=1 ursim.sh up UR5  recreates from the image, for clean controller state
#   ursim.sh down                  stops and removes the container
#   ursim.sh status                is it up, and does the dashboard answer
#
# Why it matters. URSim runs the real UR controller software, PolyScope plus the URControl core, so
# ur_rtde talks the real RTDE protocol to it. That is the difference between "the driver's calls
# look right", which is all a fake can show, and "the controller accepted them". It is also the
# only place short of a powered arm where a protective stop can be triggered on purpose.
#
# If the container exits within a minute or so of starting, hold a WSL session open. The symptoms:
#
#     docker inspect ursim   reports exit=255 with oom=false, and `docker logs` is empty
#     /ursim/polyscope.log   says "Socket connection to calibration backend failed"
#     /ursim/URControl.log   stops a few seconds in, after CONFIRM_USER_SAFETY_PARAMETERS
#
# The cause is the WSL session, not URSim. Every `wsl -e bash -lc ...` opens a session and closes it
# again; with no session left open, the distro is torn down and takes systemd, dockerd and every
# running container with it. `systemctl show docker -p NRestarts` shows the daemon is not being
# restarted after a crash: it is being stopped and brought back by socket activation. Holding one
# long-lived session open fixes it:
#
#     powershell> Start-Process wsl.exe -ArgumentList "-d","<distro>","-e","sleep","infinity" `
#                     -WindowStyle Hidden
#
# A `nohup ... &` inside a throwaway `wsl -e bash -lc` is not enough: it dies with the session it
# was meant to outlive. A real, persistent wsl.exe process is.
#
# Two things the early exit is not, both checked so nobody checks them again:
#   * Not out of memory (`oom=false`) and not out of disk.
#   * Not the calibration warning. URControl.log says "Cannot open configuration file
#     '/ursim/.urcontrol/calibration.conf'. The file is corrupt or missing." That file is absent
#     from the pristine image too, which `docker create` plus `docker cp` shows, so every healthy
#     run prints it.
#
# Remote control is required and cannot be set from here. In local mode the controller never runs a
# program sent from outside, so RTDEControlInterface times out with a message that names nothing.
# Enable it at http://localhost:6080/vnc.html under menu, Settings, System, Remote Control, then
# switch the top-right selector from Local to Remote. `URConnection.connect()` diagnoses the local
# case explicitly, so a cell that forgets gets told rather than left guessing.
#
# Ports, and why each one is published:
#   29999  dashboard server: power on, brake release, safetystatus, unlockProtectiveStop
#   30001  primary interface: URScript upload, which ur_rtde's control interface uses
#   30002  secondary interface
#   30003  realtime interface
#   30004  RTDE: the data path rtde_receive and rtde_io use
#   6080   noVNC web UI: the teach pendant in a browser, for triggering stops by hand
#   502    Modbus TCP: the UR controller's own server, see below
#   63352  the Robotiq_Grippers URCap socket, see below
#
# WSL2 forwards published ports to Windows' localhost, so ur_rtde running in the Windows Python
# reaches this at 127.0.0.1 with no extra networking.
#
# Keep the WSL VM alive as well. A Docker container is not a WSL process, so the default idle
# timeout shuts the whole VM down during a long probe on the Windows side and takes dockerd with
# it; the container then reads exit 255 and looks like a crash. `%USERPROFILE%\.wslconfig` needs
# `[wsl2] vmIdleTimeout=-1`.
#
# No volume on /ursim/programs, and that is a refusal rather than an omission. `start-ursim.sh`
# does `rm -f $URSIM_ROOT/programs` and then symlinks it to programs.<MODEL>e. A mount point cannot
# be removed, so the symlink lands inside the volume and PolyScope comes up with a broken program
# directory. Where programs must persist, mount /ursim/programs.UR5e and never /ursim/programs.
set -euo pipefail

# 502 and 63352 each earn their line.
#
#   502    URSim runs the UR controller's own Modbus TCP server, and it is a real one: it answers
#          function code 0x03, returns genuine exception codes (0x02 illegal data address, 0x03
#          illegal data value) and pairs transaction ids across consecutive round trips. That makes
#          it an independent check of a hand-rolled MBAP implementation, which is what
#          `src/robot/grippers/onrobot_modbus.py` is. There is no probe in this directory for it.
#          The registers are UR's, not OnRobot's: this validates the framing and says nothing about
#          what any address means on an OnRobot Compute Box.
#   63352  where the Robotiq_Grippers URCap would listen. Publishing the port is what exposes the
#          trap: with the port forwarded and the URCap not installed, a TCP connect succeeds and
#          the connection is dropped on the first byte. Docker forwards; nothing behind it listens.
#          So an open port here proves nothing and only a parseable reply does, which is what
#          `probe_robotiq_urcap.py` asks for.

IMAGE="${URSIM_IMAGE:-universalrobots/ursim_e-series:latest}"
NAME="${URSIM_NAME:-ursim}"

# URCaps: set `URSIM_URCAPS` to a directory of `.jar` files and they are installed headlessly.
#
#     URSIM_URCAPS=/mnt/d/dev/urcaps ./ursim.sh up UR5
#
# The image's own entrypoint does `cp -r /urcaps/*.jar /ursim/GUI/bundle/` before PolyScope starts,
# so mounting a directory there installs a URCap with no click path at all. A `.urcap` file is an
# OSGi bundle jar: rename it to `.jar`, because the entrypoint globs for that extension and a file
# named `.urcap` is silently skipped.
#
# The file itself cannot be fetched from here, and that is not a tooling gap. `Robotiq_Grippers-
# X.X.X.urcap` lives behind the 2F-85 and 2F-140 product pages on robotiq.com/support; there is no
# public direct URL, and downloading vendor software on somebody's behalf skips the licence terms
# they are meant to see. Download it once, drop it in a directory, point this variable at it.
#
# Until then, port 63352 is forwarded emptiness. See the note above: it accepts a TCP connect and
# closes on the first byte, which is exactly what "no URCap" looks like from outside.
URCAPS_DIR="${URSIM_URCAPS:-}"

cmd="${1:-status}"
model="${2:-UR5}"

# Wait for PolyScope, not for the port. A bare TCP connect to 29999 succeeds at once, because
# Docker's proxy accepts it while the controller is still booting. The honest readiness signal is
# the dashboard answering its own protocol, so this speaks `robotmode` and waits for a real reply.
# Nothing may touch the interface ports before this returns: opening 30001 to 30004 mid-boot is
# what PolyScope does not survive.
#
# The heredoc stays stdlib-only. It runs inside the WSL distro, which has neither this repository
# nor ur_rtde on its path, so `URConnection`'s dashboard methods are unreachable from here, and
# they would need a `connect()` that has not happened yet in any case.
wait_for_polyscope () {
  echo "== waiting for PolyScope to answer, not just for the port to open =="
  local i reply
  for i in $(seq 1 150); do
    reply=$(timeout 4 python3 - <<'PYEOF' 2>/dev/null
import socket, time
try:
    s = socket.create_connection(("127.0.0.1", 29999), timeout=3)
    s.settimeout(3)
    s.recv(256)
    s.sendall(b"robotmode\n")
    time.sleep(0.3)
    print(s.recv(256).decode(errors="replace").strip())
    s.close()
except Exception:
    pass
PYEOF
)
    if [ -n "$reply" ]; then
      echo "PolyScope answered after ${i}s: $reply"
      return 0
    fi
    sleep 1
  done
  echo "POLYSCOPE NEVER ANSWERED in 150s"
  docker logs --tail 20 "$NAME" || true
  return 1
}

case "$cmd" in
  up)
    # Reuse a stopped container by default: a deliberate trade-off with a sharp edge.
    #
    # Reuse does not keep remote control, which is the claim it was once made on. Across a stop and
    # a start of the same container, `/ursim/.polyscope/remotecontrol.properties` reads
    # RemoteControlEnabled=true while the dashboard `is in remote control` reads false, in every
    # sample and through power-on and brake-release. The container has no volumes at all, so the
    # file does persist in the writable layer; what does not persist is the runtime state.
    # PolyScope's remote control is two things and only the first is a setting: the feature enable,
    # which is that file and survives, and the top-right Local/Remote selector, which boots to
    # Local every time. Nothing in the dashboard protocol or in RTDE can move that selector; only a
    # person at the pendant can. So budget for enabling remote control after every start, fresh or
    # reused.
    #
    # Against reuse: the controller's float registers are part of that state.
    # `getForwardKinematics(q)` reads the TCP-offset registers, and anything that previously wrote
    # them, notably a call passing an explicit tcp_offset, makes every later q-only call return a
    # pose offset by that amount. On a fresh controller those registers are zero, which is right
    # for a bare flange. A reused container can carry a poisoned register into a measurement, and
    # the result looks like a library bug.
    #
    # Rule of thumb, now that reuse buys no remote control: URSIM_FRESH=1 whenever anything
    # kinematic is being measured, and reuse only when the container's own state is what you want
    # back (a loaded program, an installation, a safety configuration), never merely to avoid the
    # pendant.
    if [ "${URSIM_FRESH:-0}" = "1" ]; then
      echo "== URSIM_FRESH=1: recreating for clean registers; remote control needs enabling =="
      docker rm -f "$NAME" >/dev/null 2>&1 || true
    elif docker inspect "$NAME" >/dev/null 2>&1; then
      echo "== reusing the container: it carries controller state, float registers included =="
      echo "== remote control still has to be re-enabled at the pendant; reuse does not keep it =="
      docker start "$NAME" >/dev/null
      wait_for_polyscope
      docker ps --filter "name=$NAME" --format 'container: {{.Names}} / {{.Status}}'
      echo "URSIM_UP model=<reused>"
      exit 0
    fi

    echo "== pulling $IMAGE (first run only, about 2-3 GB) =="
    docker pull "$IMAGE"
    echo "== starting $NAME as ROBOT_MODEL=$model =="
    # -t is not cosmetic: the entrypoint is a foreground script whose banner ends "Press Ctrl-C to
    # exit", and a foreground script deserves a terminal.
    docker run -d -t --name "$NAME" \
      -e ROBOT_MODEL="$model" \
      -p 29999:29999 -p 30001:30001 -p 30002:30002 -p 30003:30003 -p 30004:30004 -p 6080:6080 \
      -p 502:502 -p 63352:63352 \
      ${URCAPS_DIR:+-v "$URCAPS_DIR":/urcaps} \
      "$IMAGE" >/dev/null
    wait_for_polyscope
    docker ps --filter "name=$NAME" --format 'container: {{.Names}} / {{.Status}}'
    echo "URSIM_UP model=$model"
    ;;
  down)
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    echo "URSIM_DOWN"
    ;;
  status)
    docker ps -a --filter "name=$NAME" --format '{{.Names}} / {{.Status}}' || true
    if (exec 3<>/dev/tcp/127.0.0.1/29999) 2>/dev/null; then
      echo "dashboard: port reachable, which is not proof PolyScope is up; see wait_for_polyscope"
      exec 3<&- 2>/dev/null || true
    else
      echo "dashboard: NOT reachable"
    fi
    ;;
  *)
    echo "usage: ursim.sh {up MODEL|down|status}   (URSIM_FRESH=1 forces a clean controller)" >&2
    exit 2 ;;
esac
