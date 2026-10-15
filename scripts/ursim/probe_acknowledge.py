"""Does a person clear a real protective stop through the one door, and only when they say yes? (P0i closing proof)

`src/robot/faults` is the one door to `acknowledge_fault` (P0i i4). This drives it the way a person at the cell
would, against real controller software, with the dashboard's `safetystatus` as the oracle:

  1. the arm connects through the planned arm's connect (P0i i5: the tool frame and the payload come from the tool it
     holds), and a raw `moveL` sent past our own guards on purpose provokes a real protective stop;
  2. `python -m src.robot.faults acknowledge`, answered "no" at its prompt: DECLINED, exit 1, the stop stays;
  3. the same with `--yes`, once the controller admits an unlock (about 5 s after the stop): ACKNOWLEDGED, exit 0,
     and the controller reads NORMAL;
  4. once more with `--yes`: NO_FAULT, exit 0.

A stop still standing at the end is unlocked from the dashboard, so the cell is left as it was found. The raw
``moveL`` passes every guard on purpose, so the probe runs on 127.0.0.1 alone and refuses a chain whose ``ur.ip``
is any other address before anything connects (exit 2), as ``probe_halt.py`` does.

Run (URSim up, in REMOTE control):  python scripts/ursim/probe_acknowledge.py
The UR3e:                          WILLY_PROFILE=ursim,ursim_ur3 python scripts/ursim/probe_acknowledge.py
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

# Honour an explicit chain (e.g. "ursim,ursim_ur3") so the same probe runs both models.
os.environ.setdefault("WILLY_PROFILE", "ursim")

from src.config.loader import load_robot_config  # noqa: E402
from src.robot.core import RobotVendor  # noqa: E402
from src.robot.drivers import create_arm  # noqa: E402
from src.robot.drivers.ur.arm import URRobotArm  # noqa: E402

#: The controller admits an unlock this long after a protective stop, and not before.
_UNLOCK_AFTER_S = 5.2
#: Where the raw moveL sends the TCP: over the base axis, which a UR answers with a protective stop.
#: The only controllers it is sent to: a URSim on this machine.
_LOOPBACK = ("127.0.0.1", "localhost", "::1")
_INTO_THE_BASE = [0.0, 0.0, 0.42, 0.0, 3.14, 0.0]


class Dashboard:
    """One dashboard connection, kept open so a poll costs a round trip and not a connect."""

    def __init__(self, host: str) -> None:
        self._sock = socket.create_connection((host, 29999), timeout=15)
        self._sock.settimeout(15)
        self._sock.recv(256)

    def ask(self, command: str) -> str:
        self._sock.sendall((command + "\n").encode())
        return self._sock.recv(512).decode(errors="replace").strip()

    def safety(self) -> str:
        return self.ask("safetystatus").split(":", 1)[-1].strip()

    def mode(self) -> str:
        return self.ask("robotmode").split(":", 1)[-1].strip()

    def wait_for(self, wanted: str, *, within_s: float) -> float | None:
        since = time.monotonic()
        while time.monotonic() - since < within_s:
            if self.safety() == wanted:
                return time.monotonic() - since
            time.sleep(0.05)
        return None

    def running(self) -> str:
        """Power the simulated arm on and release its brakes, as ``probe_ursim.py`` does; the robot mode it reached."""
        for command, wanted, within_s in (("power on", "IDLE", 60.0), ("brake release", "RUNNING", 90.0)):
            if self.mode() == "RUNNING":
                break
            print(f"   {command:<24} -> {self.ask(command)}")
            since = time.monotonic()
            while time.monotonic() - since < within_s and self.mode() not in (wanted, "RUNNING"):
                time.sleep(0.5)
        return self.mode()

    def close(self) -> None:
        self._sock.close()


def acknowledge(profile: str, *, yes: bool, answer: str = "") -> tuple[int, str]:
    """Run the one door as a person would; its exit code and what it printed."""
    command = [sys.executable, "-m", "src.robot.faults", "acknowledge", "--profile", profile]
    if yes:
        command.append("--yes")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    done = subprocess.run(command, input=answer, capture_output=True, encoding="utf-8", errors="replace",
                          timeout=180, env=env)
    return done.returncode, done.stdout


def main() -> int:
    profile = os.environ["WILLY_PROFILE"]
    cfg = load_robot_config()
    host = str(cfg.arm.controller)
    if host not in _LOOPBACK:
        print(f"refused: the chain {profile!r} points at {host}, not the loopback. This probe sends a raw moveL past "
              "every guard, so it runs on a URSim at 127.0.0.1 alone.")
        return 2
    results: list[tuple[str, bool]] = []
    dash = Dashboard(host)
    try:
        print(f"=== 0. the controller at {host}, profile {profile}")
        print(f"   robot mode {dash.running()}, safety {dash.safety()}")

        print("\n=== 1. connect, then a real protective stop")
        arm = create_arm(RobotVendor.from_string(cfg.vendor), config=cfg)
        if not isinstance(arm, URRobotArm):
            raise SystemExit(f"vendor {cfg.vendor!r} builds {type(arm).__name__}; this probe reads UR-only seams")
        arm.connect()
        print(f"   connected, {arm.mounted_tool.render()}")
        try:
            # Past our own guards on purpose: the stop is the state under test, not a motion we would ever send.
            ctrl = arm._conn._ctrl
            if ctrl is None:
                raise RuntimeError("the RTDE control interface is not open")
            ctrl.moveL(_INTO_THE_BASE, 0.9, 1.2)
        except Exception as exc:  # noqa: BLE001
            print(f"   raw moveL: {str(exc)[:80]}")
        took = dash.wait_for("PROTECTIVE_STOP", within_s=5.0)
        stopped_at = time.monotonic()
        try:
            arm.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(f"   disconnect: {str(exc)[:80]}")
        results.append(("a real protective stop", took is not None))
        print(f"   safety {dash.safety()} ({'stopped' if took is not None else 'NO STOP'})")
        if took is None:
            return 1

        print("\n=== 2. the door, answered 'no' at its prompt")
        code, out = acknowledge(profile, yes=False, answer="no\n")
        print("   " + out.strip().replace("\n", "\n   "))
        safety = dash.safety()
        ok = code == 1 and "DECLINED" in out and safety == "PROTECTIVE_STOP"
        results.append(("declined, exit 1, the stop stays", ok))
        print(f"   exit {code}, safety {safety}")

        wait = _UNLOCK_AFTER_S - (time.monotonic() - stopped_at)
        if wait > 0:
            time.sleep(wait)
        print("\n=== 3. the door, with --yes")
        code, out = acknowledge(profile, yes=True)
        print("   " + out.strip().replace("\n", "\n   "))
        normal = dash.wait_for("NORMAL", within_s=10.0)
        ok = code == 0 and "ACKNOWLEDGED" in out and normal is not None
        results.append(("acknowledged, exit 0, NORMAL", ok))
        print(f"   exit {code}, safety {dash.safety()}")

        print("\n=== 4. the door again, with nothing to clear")
        code, out = acknowledge(profile, yes=True)
        print("   " + out.strip().replace("\n", "\n   "))
        ok = code == 0 and "NO_FAULT" in out
        results.append(("no fault, exit 0", ok))
        print(f"   exit {code}, safety {dash.safety()}")
    finally:
        if dash.safety() == "PROTECTIVE_STOP":
            time.sleep(_UNLOCK_AFTER_S)
            for command in ("close safety popup", "unlock protective stop"):
                print(f"   left stopped; {command:<24} -> {dash.ask(command)}")
        dash.close()

    print()
    for name, ok in results:
        print(f"   {'OK  ' if ok else 'FAIL'} {name}")
    passed = sum(ok for _, ok in results)
    print(f"{passed} of {len(results)} as expected")
    print("ACKNOWLEDGE_PROBE_DONE")
    return 0 if passed == len(results) == 4 else 1


if __name__ == "__main__":
    sys.exit(main())
